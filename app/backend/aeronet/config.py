"""Deployment settings, all from environment variables (read when asked, so tests can change them).

AERONET_WITHHOLD_FVM_DATA=1   Public deployments: never serve NNfoil's finite-volume Cp files or curves (they have no licence for
                              redistribution). The numbers computed from them (errors in results/summary.csv) are still shown.
AERONET_CORS_ORIGINS=a,b      Origins allowed to call the API from another site. Not needed when one server serves page and API.
AERONET_MAX_HEAVY=2           How many expensive requests (optimiser, viscous check) may run at the same time.
AERONET_HEAVY_PER_MIN=30      Expensive requests per minute and client address.
AERONET_RESULTS_DIR, AERONET_FVM_DIR   Where the PINN results and the FVM Cp files are (see pinn_data.py).
"""

from __future__ import annotations

import os

DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, lo: int = 1) -> int:
    try:
        return max(lo, int(os.environ.get(name, default)))
    except ValueError:
        return default


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
