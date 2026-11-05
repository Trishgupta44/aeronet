"""Verification checks for NNfoil (promise P7): mapping, normals, freestream residual, force convention."""

import math

import numpy as np
import pytest
import torch

import metrics
import NNfoil_Euler2D as nf

ALPHA = 4.0
MA = 0.4


@pytest.fixture(scope="module")
def nn(mesh_50x25):
    xx, yy = mesh_50x25
    return nf.NNfoil([2, 16, 16, 4], torch.device("cpu"), xx, yy, Ma=MA, alpha_deg=ALPHA)


class Freestream(torch.nn.Module):
    """Constant (u, v, p, rho) = freestream; the 0*x term gives autograd a path to the input."""

    def __init__(self, nn_):
        super().__init__()
        self.c = torch.tensor([[nn_.far["u"].item(), nn_.far["v"].item(), nn_.far["p"].item(), 1.0]])

    def forward(self, x):
        return self.c.expand(x.shape[0], 4) + 0 * x.sum(1, keepdim=True)


def test_surface_mapping_wall_is_mesh_column_0(nn, mesh_50x25):
    xx, yy = mesh_50x25
    ref = np.c_[xx[:, 0], yy[:, 0]]
    assert np.abs(nn.wall["X"].numpy() - ref).max() < 1e-6


def test_outward_normals_for_every_face(nn):
    X, Xf, nor = nn.wall["X"][:-1], nn.wall["X_f"], nn.wall["nor_f"]
    centroid = X.mean(0, keepdim=True)
    dot = ((Xf - centroid) * nor).sum(1)
    assert (dot > 0).float().mean().item() == 1.0


def test_freestream_has_zero_residual_and_forces(mesh_50x25):
    xx, yy = mesh_50x25
    net = nf.NNfoil([2, 16, 16, 4], torch.device("cpu"), xx, yy, Ma=MA, alpha_deg=ALPHA)
    net.model = Freestream(net)
    net.batch_ind = np.arange(net.XE.shape[0])
    with torch.enable_grad():
        assert net.Msef().item() == 0.0
    # far-field part of Mseb (same expression as in Mseb)
    with torch.no_grad():
        pred = net.model(net.far["XE"])
        far = (
            ((pred[:, 2:3] - net.far["p"]) ** 2).mean()
            + ((pred[:, 0:1] - net.far["u"]) ** 2).mean()
            + ((pred[:, 1:2] - net.far["v"]) ** 2).mean()
            + ((pred[:, 3:4] - net.far["rho"]) ** 2).mean()
        )
        assert far.item() == 0.0
        # the wall part is NOT zero: the freestream is not tangent to the wall (expected)
        assert net.Mseb().item() > 1e-6
    cl, cd = net.evaluate_wall()
    assert abs(cl) < 1e-6 and abs(cd) < 1e-6


def test_calculate_force_matches_metrics_convention(nn, mesh_50x25):
    xx, yy = mesh_50x25
    n_faces = xx.shape[0] - 1
    cp_f = np.sin(np.arange(n_faces, dtype=float))
    nn.wall["Cp_f"] = torch.tensor(cp_f, dtype=torch.float32).reshape(-1, 1)
    nn.Calculate_Force()
    cl, cd = metrics.forces_from_faces(xx[:, 0], yy[:, 0], cp_f, ALPHA)
    assert math.isclose(float(nn.Cl), cl, abs_tol=1e-5)
    assert math.isclose(float(nn.Cd), cd, abs_tol=1e-5)
