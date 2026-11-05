"""Baseline rows (experiment B0) of results/summary.csv: FVM coarse vs fine, panel vs FVM, thin-airfoil Cl.

    research/.venv/Scripts/python research/baselines.py [--out results/summary.csv]

Every number is computed here from the bundled FVM files; nothing is typed in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (HERE, REPO / "app" / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import metrics  # noqa: E402
from aeronet import compare, geometry, panel  # noqa: E402

FVM = REPO / "NNfoil" / "FVM_Cp"
FVM2412 = REPO / "NNfoil" / "FVM_Cp_NACA2412"
MACH_0012, ALPHA_0012 = 0.4, 4
MACH_2412 = 0.25  # fitted from the FVM Cp files (roadmap 1a, item 2); confirm from the paper
NOTE_2412 = "Mach fitted from Cp, confirm from paper"
SWEEP_2412 = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
MESHES = [("50x25", "50_25"), ("100x50", "100_50"), ("200x100", "200_100"), ("400x200", "400_200")]


def _r(v, nd=6):
    return None if v is None else round(float(v), nd)


def _case(shape, alpha):
    return f"{shape}_a{float(alpha):g}"


def _row(**kw):
    base = {"experiment": "B0", "source": "measured", "notes": ""}
    base.update(kw)
    for k in ("rms_cp", "max_cp", "cl", "cl_ref", "cd"):
        if k in base:
            base[k] = _r(base[k])
    return base


def build_rows() -> list[dict]:
    rows = []

    # (1) FVM coarse meshes vs the 800x400 file, NACA 0012, Mach 0.4, 4 deg
    ref_path = FVM / "NACA0012_800_400_Cp.dat"
    ref = metrics.load_fvm(ref_path)
    cl_ref, _ = metrics.fvm_forces(ref_path, ALPHA_0012)
    for label, tag in MESHES:
        path = FVM / f"NACA0012_{tag}_Cp.dat"
        pts = metrics.load_fvm(path)
        err = metrics.cp_error(pts[:, 0], pts[:, 1], pts[:, 2], ref)
        cl, cd = metrics.fvm_forces(path, ALPHA_0012)
        rows.append(_row(case=_case("NACA0012", ALPHA_0012), method="fvm", mesh=label, mach=MACH_0012,
                         alpha_deg=ALPHA_0012, rms_cp=err["rms"], max_cp=err["max"], cl=cl, cl_ref=cl_ref, cd=cd,
                         notes="reference: FVM 800x400"))

    # (2) panel method, NACA 0012, vs the 800x400 file
    sol = panel.solve("0012", ALPHA_0012, MACH_0012, 100)
    err = metrics.cp_error(sol.geom.cx, sol.geom.cy, sol.cp, ref)
    rows.append(_row(case=_case("NACA0012", ALPHA_0012), method="panel", mesh=None, mach=MACH_0012,
                     alpha_deg=ALPHA_0012, rms_cp=err["rms"], max_cp=err["max"], cl=sol.cl, cl_ref=cl_ref, cd=sol.cd,
                     notes="reference: FVM 800x400; 200 panels"))

    # (3) panel method, NACA 2412, one row per AoA file at the fitted Mach 0.25
    for aoa in SWEEP_2412:
        path = FVM2412 / f"NACA2412_800_400_AoA{aoa}_Cp.dat"
        pts = metrics.load_fvm(path)
        sol = panel.solve("2412", aoa, MACH_2412, 100)
        err = metrics.cp_error(sol.geom.cx, sol.geom.cy, sol.cp, pts)
        cl_f, _ = metrics.fvm_forces(path, aoa)
        rows.append(_row(case=_case("NACA2412", aoa), method="panel", mesh=None, mach=MACH_2412, alpha_deg=aoa,
                         rms_cp=err["rms"], max_cp=err["max"], cl=sol.cl, cl_ref=cl_f, cd=sol.cd, source="fitted",
                         notes=NOTE_2412))

    # (4) thin-airfoil Cl at 4 deg (a computed value, labelled measured)
    for code, mach, ref_file, note in [
        ("0012", MACH_0012, ref_path, "Prandtl-Glauert corrected; reference: FVM 800x400"),
        ("2412", MACH_2412, FVM2412 / "NACA2412_800_400_AoA4_Cp.dat", f"{NOTE_2412}; reference: FVM 800x400"),
    ]:
        cl_t = compare.thin_airfoil_cl(geometry.naca4(code), 4, mach)
        cl_f, _ = metrics.fvm_forces(ref_file, 4)
        rows.append(_row(case=_case(f"NACA{code}", 4), method="thin_airfoil", mesh=None, mach=mach, alpha_deg=4,
                         cl=cl_t, cl_ref=cl_f, notes=note))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/summary.csv")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_rows()
    for r in rows:
        metrics.append_summary(r, a.out)
    print(f"wrote {len(rows)} B0 rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
