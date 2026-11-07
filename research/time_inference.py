"""Timing for the E4 speed table (Plan C, task C4): PINN inference, the panel method and NeuralFoil.

    research/.venv/Scripts/python research/time_inference.py <run_dir> [--calls 100] [--threads 8]
    research/.venv/Scripts/python research/time_inference.py --e4 [--out results/e4_timing.json]

The first form prints ``{"infer_ms","infer_ms_std","n_points","threads"}`` (plus the wall-only time) for one run.
The second form runs the whole E4 measurement ONE TIMING AT A TIME (never in parallel) and writes a JSON that
``experiments.py collect e4`` turns into ``results/summary.csv`` rows and ``results/e4_speed.md``.

Run it on an idle CPU: other work (training, tests) spoils the numbers. Every number is a mean over ``--calls`` calls after
a warm-up, with the standard deviation (n - 1) and the median / min / max next to it.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "app" / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CALLS, WARMUP = 100, 10
E4_THREADS = (1, 8)
# the runs whose networks are timed (see experiments.E4_PINN_RUNS for the train_s side)
E4_RUN_NAMES = ["naca0012_50x25_a4_m04", "naca0012_50x25_a4_m04_nopos", "naca0012_100x50_a4_m04",
                "naca0012_100x50_a4_m04_gs", "naca0012_100x50_a4_m04_gs_nopos", "naca0012_200x100_a4_m04_gs_nopos"]


# ---------------------------------------------------------------- helpers
def summarize(ms) -> dict:
    """Mean, sample standard deviation (n - 1), median, min, max and n of a list of times in milliseconds."""
    a = np.asarray(ms, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def time_callable(fn, calls=CALLS, warmup=WARMUP) -> list[float]:
    """``warmup`` untimed calls, then ``calls`` calls timed one by one with ``perf_counter``; returns milliseconds."""
    for _ in range(int(warmup)):
        fn()
    out = []
    for _ in range(int(calls)):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def selig_coords(code="0012", n_half=70) -> np.ndarray:
    """NACA coordinates in the app frame (x 0..1) in Selig order: trailing edge, upper surface, leading edge, lower
    surface, trailing edge again (the closing node repeats the first one)."""
    from aeronet import geometry

    af = geometry.naca4(code, n_half)
    x, y = np.r_[af.x, af.x[0]], np.r_[af.y, af.y[0]]
    return np.c_[x, y]


def cpu_name() -> str:
    """CPU model from ``Get-CimInstance Win32_Processor`` (Windows), else ``platform.processor()``."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        if out:
            return out
    except Exception:  # noqa: BLE001 - any failure falls back to the generic name
        pass
    return platform.processor() or "unknown"


def hardware() -> dict:
    import os

    import torch

    try:
        import importlib.metadata as md

        nf_version = md.version("neuralfoil")
    except Exception:  # noqa: BLE001
        nf_version = None
    return {"cpu": cpu_name(), "logical_threads": os.cpu_count(), "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(), "neuralfoil": nf_version}


def cpu_load(samples=3, interval=3) -> list[int]:
    """Total CPU load (percent) from the Windows performance counter, e.g. ``[4, 7, 5]``; empty if not available."""
    cmd = (f'(Get-Counter "\\Processor(_Total)\\% Processor Time" -SampleInterval {int(interval)} '
           f'-MaxSamples {int(samples)}).CounterSamples | % {{[int]$_.CookedValue}}')
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                             timeout=120).stdout.split()
        return [int(v) for v in out]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------- PINN
def time_pinn(run_dir, calls=CALLS, warmup=WARMUP, threads=(8,), nn=None) -> dict:
    """Full-field forward pass ``model(XE)`` and wall-only ``evaluate_wall()`` of a trained run, per thread count.

    ``field`` gives u, v, p, rho at every mesh node (the whole flow field); ``wall`` is ``evaluate_wall`` (Cp on the
    wall nodes and faces plus Cl and Cd). Both run under ``torch.no_grad()``.
    """
    import torch

    import runs

    nn = nn if nn is not None else runs.build_model(run_dir, threads=int(threads[0]))
    out = {"n_points": int(nn.XE.shape[0]), "threads": {}}
    for th in threads:
        torch.set_num_threads(int(th))

        def field():
            with torch.no_grad():
                nn.model(nn.XE)

        out["threads"][str(int(th))] = {"field": summarize(time_callable(field, calls, warmup)),
                                        "wall": summarize(time_callable(nn.evaluate_wall, calls, warmup))}
    return out


# ---------------------------------------------------------------- panel method and NeuralFoil
def time_panel(calls=CALLS, warmup=WARMUP, width=170) -> dict:
    """``panel.solve("0012", 4, 0.4, 70)`` alone, and the same plus ``field.field_grid(sol, width)``."""
    from aeronet import field, panel

    solve = summarize(time_callable(lambda: panel.solve("0012", 4, 0.4, 70), calls, warmup))
    both = summarize(time_callable(lambda: field.field_grid(panel.solve("0012", 4, 0.4, 70), width), calls, warmup))
    return {"solve": solve, "solve_field170": both, "n_panels": int(panel.solve("0012", 4, 0.4, 70).n),
            "field_width": int(width)}


def time_neuralfoil(calls=CALLS, warmup=WARMUP, model_size="large") -> dict:
    """``neuralfoil.get_aero_from_coordinates`` for NACA 0012 at 4 degrees, Re 6e6 (Kulfan fit included in the time).

    NeuralFoil is viscous and incompressible. Returns ``{"error": ...}`` when the import or the call fails.
    """
    try:
        import neuralfoil as nf

        xy = selig_coords("0012", 70)

        def call():
            return nf.get_aero_from_coordinates(coordinates=xy, alpha=4.0, Re=6e6, model_size=model_size)

        res = call()
        ms = time_callable(call, calls, warmup)
        pick = lambda k: float(np.asarray(res[k]).ravel()[0])  # noqa: E731
        import importlib.metadata as md

        return {"time": summarize(ms), "cl": pick("CL"), "cd": pick("CD"), "confidence": pick("analysis_confidence"),
                "model_size": model_size, "version": md.version("neuralfoil")}
    except Exception as e:  # noqa: BLE001 - recorded as "not measured", never guessed
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- the E4 measurement
def measure_e4(calls=CALLS, warmup=WARMUP, runs_root=None, log=print) -> dict:
    """Every E4 timing, one after another. ``cpu_load_before`` holds the CPU load seen before each block."""
    import runs

    runs_root = Path(runs_root) if runs_root is not None else runs.RUNS
    res = {"hardware": hardware(), "calls": calls, "warmup": warmup, "cpu_load_before": {}, "pinn": {}}
    t_start = time.time()
    for name in E4_RUN_NAMES:
        res["cpu_load_before"][name] = cpu_load()
        log(f"{name}: cpu load {res['cpu_load_before'][name]}")
        res["pinn"][name] = time_pinn(runs_root / name, calls, warmup, E4_THREADS)
        t = res["pinn"][name]["threads"]
        log(f"  field 1t {t['1']['field']['mean']:.3f} ms  8t {t['8']['field']['mean']:.3f} ms  "
            f"(max/median {t['8']['field']['max'] / t['8']['field']['median']:.1f})")
    res["cpu_load_before"]["panel"] = cpu_load()
    res["panel"] = time_panel(calls, warmup)
    log(f"panel solve {res['panel']['solve']['mean']:.3f} ms, +field(170) {res['panel']['solve_field170']['mean']:.2f} ms")
    res["cpu_load_before"]["neuralfoil"] = cpu_load()
    res["neuralfoil"] = time_neuralfoil(calls, warmup)
    log(f"neuralfoil {res['neuralfoil'].get('time', res['neuralfoil'])}")
    res["wall_clock_s"] = time.time() - t_start
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="?", help="a run folder (time its network)")
    ap.add_argument("--calls", type=int, default=CALLS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--e4", action="store_true", help="run the whole E4 measurement and write --out")
    ap.add_argument("--out", default="results/e4_timing.json")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.e4:
        res = measure_e4(a.calls, a.warmup)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf8")
        print(f"wrote {a.out}")
        return 0
    if not a.run_dir:
        ap.error("give a run folder or --e4")
    r = time_pinn(a.run_dir, a.calls, a.warmup, (a.threads,))
    f = r["threads"][str(a.threads)]
    print(json.dumps({"infer_ms": f["field"]["mean"], "infer_ms_std": f["field"]["sd"], "n_points": r["n_points"],
                      "threads": a.threads, "wall_ms": f["wall"]["mean"], "wall_ms_std": f["wall"]["sd"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
