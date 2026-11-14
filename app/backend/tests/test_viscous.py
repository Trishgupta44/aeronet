"""Optional NeuralFoil estimate (aeronet/neural.py, POST /api/viscous)."""

import pytest
from fastapi.testclient import TestClient

from aeronet import neural
from aeronet.main import app

client = TestClient(app)
needs_nf = pytest.mark.skipif(not neural.available(), reason="neuralfoil not installed")


def test_503_with_a_clear_message_when_neuralfoil_is_missing(monkeypatch):
    monkeypatch.setattr(neural, "_nf", None)
    r = client.post("/api/viscous", json={"code": "0012", "alpha": 4, "re": 1e6})
    assert r.status_code == 503 and "not installed" in r.json()["detail"]


def test_bad_code_and_bad_reynolds_are_rejected():
    assert client.post("/api/viscous", json={"code": "abc", "alpha": 4, "re": 1e6}).status_code == 422
    assert client.post("/api/viscous", json={"code": "0012", "alpha": 4, "re": 10}).status_code == 422


@needs_nf
def test_naca0012_viscous_result_is_sensible_next_to_the_inviscid_one():
    r = client.post("/api/viscous", json={"code": "0012", "alpha": 4, "re": 1e6})
    assert r.status_code == 200
    d = r.json()
    assert "viscous" in d["label"] and d["inviscid"]["mach"] == 0.0
    assert d["cl"] < d["inviscid"]["cl"]            # boundary layer lowers the lift
    assert 0.003 < d["cd"] < 0.02                   # real drag, unlike the inviscid cd (about 0)
    assert abs(d["inviscid"]["cd"]) < 0.01
    assert 0.0 <= d["confidence"] <= 1.0 and d["call_ms"] > 0
    assert len(d["polar"]["alpha"]) == len(d["polar"]["cl"]) == len(d["polar"]["cl_inviscid"]) == 21


@needs_nf
def test_higher_reynolds_number_gives_less_drag():
    cd = {re: client.post("/api/viscous", json={"code": "0012", "alpha": 2, "re": re}).json()["cd"] for re in (1e5, 1e7)}
    assert cd[1e7] < cd[1e5]
