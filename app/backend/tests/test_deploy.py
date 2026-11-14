"""Deployment settings: the public build (no FVM data), CORS origins, and the limits on the expensive endpoints."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from aeronet import config, limits, optimize, pinn_data
from aeronet.main import app

HEADER = "experiment,case,method,mesh,mach,alpha_deg,rms_cp,max_cp,cl,cl_ref,cd,train_s,infer_ms,source,notes\n"
client = TestClient(app)


@pytest.fixture
def public(tmp_path, monkeypatch):
    """A results folder with a PINN run and NO FVM folder, served with the public switch on."""
    res = tmp_path / "results"
    (res / "runs" / "naca0012_100x50_a4_m04").mkdir(parents=True)
    (res / "summary.csv").write_text(
        HEADER
        + "B0,NACA0012_a4,fvm,100x50,0.4,4,0.045985,0.260987,0.480267,0.519329,0.01,,,measured,ref\n"
        + "E2,NACA0012_a4,pinn,100x50,0.4,4,0.014451,0.082595,0.516286,0.519329,0.0006,577.6,,measured,final round; rounds=20; failed_rounds=0\n"
    )
    x = np.linspace(0.05, 0.95, 10).tolist()
    (res / "runs" / "naca0012_100x50_a4_m04" / "snapshots.json").write_text(
        json.dumps({"x": x, "y": [0.03] * 5 + [-0.03] * 5, "rounds": [{"round": 1, "cp": [0.1] * 10}]})
    )
    monkeypatch.setenv("AERONET_RESULTS_DIR", str(res))
    monkeypatch.setenv("AERONET_FVM_DIR", str(tmp_path / "no_fvm_here"))
    monkeypatch.setenv("AERONET_WITHHOLD_FVM_DATA", "1")


def test_public_build_serves_numbers_but_no_fvm_curves(public):
    d = client.get("/api/showdown").json()
    assert d["fvm_curves_withheld"] is True and d["reference"]["curve"] is None
    m = next(m for m in d["meshes"] if m["mesh"] == "100x50")
    assert m["fvm"]["rms"] == 0.045985 and m["fvm"]["curve"] is None  # the error number stays, the curve does not
    assert m["pinn"][0]["curve"]["cp"] == [0.1] * 10                    # our own PINN result stays


def test_public_build_hides_the_bundled_fvm_sample(public):
    assert client.get("/api/reference/sample").status_code == 404
    caps = client.get("/api/capabilities").json()
    assert caps["fvm_sample"] is False and caps["fvm_curves"] is False and caps["public_build"] is True


def test_default_build_keeps_the_sample_and_the_curves(monkeypatch):
    monkeypatch.delenv("AERONET_WITHHOLD_FVM_DATA", raising=False)
    caps = client.get("/api/capabilities").json()
    assert caps["public_build"] is False and caps["fvm_curves"] is True
    assert caps["fvm_sample"] is True  # the sample file ships with the repository
    assert client.get("/api/reference/sample").status_code == 200


def test_cors_origins_from_the_environment(monkeypatch):
    monkeypatch.setenv("AERONET_CORS_ORIGINS", "https://demo.example.org/, https://other.example.org")
    assert config.cors_origins() == config.DEV_ORIGINS + ["https://demo.example.org", "https://other.example.org"]
    monkeypatch.delenv("AERONET_CORS_ORIGINS")
    assert config.cors_origins() == config.DEV_ORIGINS


def test_bad_numbers_in_the_environment_fall_back_to_the_defaults(monkeypatch):
    monkeypatch.setenv("AERONET_MAX_HEAVY", "many")
    monkeypatch.setenv("AERONET_HEAVY_PER_MIN", "0")
    assert config.max_heavy() == 2 and config.heavy_per_minute() == 1  # not a number: default; below the minimum: the minimum


def _request(host="203.0.113.7"):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def test_per_address_limit_answers_429(monkeypatch):
    monkeypatch.setattr(optimize, "optimise", lambda *a, **k: {"ok": True})
    monkeypatch.setenv("AERONET_HEAVY_PER_MIN", "2")
    body = {"target_cl": 0.5, "min_thickness": 0.1}
    assert [client.post("/api/optimize", json=body).status_code for _ in range(3)] == [200, 200, 429]
    assert "wait a minute" in client.post("/api/optimize", json=body).json()["detail"]


def test_concurrency_limit_answers_503_and_frees_the_slot(monkeypatch):
    monkeypatch.setenv("AERONET_MAX_HEAVY", "1")
    first = limits.heavy_guard(_request("198.51.100.1"))
    next(first)  # takes the only slot
    second = limits.heavy_guard(_request("198.51.100.2"))
    with pytest.raises(HTTPException) as busy:
        next(second)
    assert busy.value.status_code == 503 and "busy" in busy.value.detail
    first.close()  # the first request finishes
    third = limits.heavy_guard(_request("198.51.100.3"))
    next(third)
    third.close()


def test_a_refused_request_does_not_use_up_the_budget_of_the_address(monkeypatch):
    monkeypatch.setenv("AERONET_MAX_HEAVY", "1")
    monkeypatch.setenv("AERONET_HEAVY_PER_MIN", "2")
    holder = limits.heavy_guard(_request("198.51.100.9"))
    next(holder)
    refused = limits.heavy_guard(_request("203.0.113.5"))
    with pytest.raises(HTTPException):
        next(refused)
    holder.close()
    ok = limits.heavy_guard(_request("203.0.113.5"))
    next(ok)  # still has its full allowance
    ok.close()
