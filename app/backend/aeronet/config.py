"""Deployment settings, all from environment variables (read when asked, so tests can change them).

AERONET_WITHHOLD_FVM_DATA=1   Public deployments: never serve NNfoil's finite-volume (FVM) Cp data (it has no licence for redistribution):
                              not the bundled sample, not the ``reference`` / ``fvm_same_mesh`` points of the exported PINN cases, not the
                              showdown curves. The numbers computed from them (errors, Cl, ``rms_by_round``) are still shown. The image of a
                              public deployment does not even contain the curves: it is built from a stripped copy of the data
                              (deploy/make_data.py), and this switch is the second line of defence if the full data is present.
AERONET_DATA_DIR=<folder>     Where the exported data is (``pinn_cases/``, ``results_summary.json``, ``seed_runs.json``, ``rounds_e1.json``).
                              Default: app/backend/data. Read once, when the server starts. The public build points it at the stripped copy.
AERONET_CORS_ORIGINS=a,b      Origins allowed to call the API from another site. Not needed when one server serves page and API.
AERONET_MAX_HEAVY=2           How many expensive requests (optimiser, viscous check) may run at the same time.
AERONET_HEAVY_PER_MIN=30      Expensive requests per minute and client address.
"""

from __future__ import annotations

import os
from pathlib import Path

DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, lo: int = 1) -> int:
    try:
        return max(lo, int(os.environ.get(name, default)))
    except ValueError:
        return default


def data_dir(default: Path) -> Path:
    """``AERONET_DATA_DIR`` if set, else ``default``."""
    raw = os.environ.get("AERONET_DATA_DIR", "").strip()
    return Path(raw).resolve() if raw else default


def withhold_fvm_data() -> bool:
    return _flag("AERONET_WITHHOLD_FVM_DATA")


def cors_origins() -> list[str]:
    raw = os.environ.get("AERONET_CORS_ORIGINS", "")
    extra = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return DEV_ORIGINS + extra


def max_heavy() -> int:
    return _int("AERONET_MAX_HEAVY", 2)


def heavy_per_minute() -> int:
    return _int("AERONET_HEAVY_PER_MIN", 30)
