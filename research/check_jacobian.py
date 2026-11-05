"""Finding 10: wrong-sign Jacobian at the trailing-edge seam of the bundled meshes.

NNfoil.Mesh() calls util.jacobian_trans(..., boundary=(1,1)), i.e. one-sided stencils in xi at rows 0 and N. But xi is
periodic (row N is row 0 repeated), so at the trailing-edge seam the Jacobian is wrong. util.jacobian_trans_periodic
(used by NNfoil(periodic_jacobian=True)) differences periodically in xi instead.

For each bundled mesh this counts cells whose det J has the opposite sign to the majority, both ways, and also
reports how the J^2 loss weight (Msef weights each residual by J^2 / sum J^2) is spread over the eta rows.
Every number is measured; the CSV is results/finding10_jacobian.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "NNfoil", REPO / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import metrics  # noqa: E402
from util import jacobian_trans, jacobian_trans_periodic  # noqa: E402

MESHES = ["NACA0012_50_25", "NACA0012_100_50", "NACA0012_200_100", "NACA2412_200_100",
          "Airfoil1_200_100", "Airfoil2_200_100", "Airfoil3_200_100", "Airfoil4_200_100", "Airfoil5_200_100"]
COLUMNS = ["mesh", "n_xi", "n_eta", "majority_sign", "upstream_wrong_cells", "upstream_wrong_xi_rows",
           "upstream_wrong_eta_cols", "periodic_wrong_cells", "periodic_wrong_xi_rows", "upstream_max_abs_jac_inv",
           "periodic_max_abs_jac_inv", "j2_share_inner_quarter_eta", "j2_share_outer_quarter_eta", "source"]
NODES = torch.arange(-3, 4, 1, dtype=torch.long)  # the stencil NNfoil.Mesh() uses


def wrong_sign(J):
    """(majority sign, count of cells with the other sign, sorted xi rows, sorted eta columns containing them)."""
    maj = torch.sign(J).median()
    bad = (torch.sign(J) != maj).nonzero().tolist()
    return int(maj.item()), len(bad), sorted({i for i, _ in bad}), sorted({j for _, j in bad})


def analyse(mesh_path):
    xx, yy = metrics.load_mesh(mesh_path)  # closed ring (N+1, M), NNfoil frame
    X, Y = torch.tensor(xx).double(), torch.tensor(yy).double()
    jac_up, inv_up = jacobian_trans(X, Y, NODES, boundary=(1, 1))
    jac_pe, inv_pe = jacobian_trans_periodic(X, Y, NODES)
    j_up, j_pe = torch.det(jac_up), torch.det(jac_pe)
    maj, n_up, rows_up, cols_up = wrong_sign(j_up)
    _, n_pe, rows_pe, _ = wrong_sign(j_pe)
    w = j_up**2
    M = w.shape[1]
    q = max(1, M // 4)
    return {"n_xi": X.shape[0], "n_eta": M, "majority_sign": maj, "upstream_wrong_cells": n_up,
            "upstream_wrong_xi_rows": " ".join(map(str, rows_up)),
            "upstream_wrong_eta_cols": " ".join(map(str, cols_up)), "periodic_wrong_cells": n_pe,
            "periodic_wrong_xi_rows": " ".join(map(str, rows_pe)),
            "upstream_max_abs_jac_inv": float(inv_up.abs().max()), "periodic_max_abs_jac_inv": float(inv_pe.abs().max()),
            "j2_share_inner_quarter_eta": float(w[:, :q].sum() / w.sum()),
            "j2_share_outer_quarter_eta": float(w[:, M - q:].sum() / w.sum()), "source": "measured"}


def run(mesh_dir=None):
    mesh_dir = Path(mesh_dir) if mesh_dir else REPO / "NNfoil" / "mesh"
    return [{"mesh": name, **analyse(mesh_dir / f"{name}.x")} for name in MESHES]


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    rows = run()
    write_csv(rows, REPO / "results" / "finding10_jacobian.csv")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
