"""Evidence tests behind docs/verification.md: Euler residual forms, finding 3 (mesh generator), finding 10 (Jacobian)."""

import math

import numpy as np
import pytest
import torch

import check_jacobian
import check_mesh_smoothing
import metrics
import NNfoil_Euler2D as nf
from util import jacobian_trans

GAMMA = 1.4


# ---------------------------------------------------------------- Euler residuals (e1-e4 in Msef)

N_XI, N_ETA, R_IN, R_OUT = 64, 30, 0.5, 6.0
_Q = (R_OUT / R_IN) ** (1.0 / (N_ETA - 1))


def _polar(xi, eta):
    """Analytic polar mesh: xi runs round (period N_XI), eta runs out geometrically."""
    th = 2 * math.pi * xi / N_XI
    r = R_IN * _Q**eta
    return r * torch.cos(th), r * torch.sin(th)


def _fields(x, y):
    """Smooth manufactured (u, v, p, rho). They are NOT an Euler solution, so every residual is non-zero."""
    u = 1 + 0.3 * torch.sin(x) * torch.cos(y)
    v = 0.2 * torch.cos(x) + 0.1 * y / (1 + 0.1 * x * x)
    p = 3 + 0.5 * torch.cos(x + y)
    rho = 1 + 0.2 * torch.sin(y) + 0.1 * torch.cos(x)
    return u, v, p, rho


class _Manufactured(torch.nn.Module):
    def forward(self, xe):
        x, y = _polar(xe[:, 0], xe[:, 1])
        return torch.stack(_fields(x, y), dim=1)


def _grad(f, w):
    return torch.autograd.grad(f.sum(), w, retain_graph=True)[0]


def _reference(nn_, e4_gamma_factor=GAMMA):
    """Msef computed independently: derivatives in (x, y) by autograd, residuals from the CONSERVATIVE steady Euler
    equations (mass, x-momentum, y-momentum, total enthalpy flux), converted to the primitive forms by identities.

        C1 = div(rho V)                      = e1
        Cx = d(rho u^2 + p)/dx + d(rho u v)/dy = u e1 + rho e2
        Cy = d(rho u v)/dx + d(rho v^2 + p)/dy = v e1 + rho e3
        CE = div((rho E + p) V)              = |V|^2/2 e1 + rho (u e2 + v e3) + e4/(gamma-1)
    """
    xi, eta = nn_.XE[:, 0].double(), nn_.XE[:, 1].double()
    x, y = (t.detach().requires_grad_(True) for t in _polar(xi, eta))
    u, v, p, rho = _fields(x, y)
    energy = p / ((GAMMA - 1) * rho) + 0.5 * (u * u + v * v)
    c1 = _grad(rho * u, x) + _grad(rho * v, y)
    cx = _grad(rho * u * u + p, x) + _grad(rho * u * v, y)
    cy = _grad(rho * u * v, x) + _grad(rho * v * v + p, y)
    ce = _grad((rho * energy + p) * u, x) + _grad((rho * energy + p) * v, y)
    e1 = c1
    e2 = (cx - u * c1) / rho
    e3 = (cy - v * c1) / rho
    e4 = (GAMMA - 1) * (ce + 0.5 * (u * u + v * v) * c1 - u * cx - v * cy)
    w = nn_.J.double().reshape(-1) ** 2
    msef = ((e1**2 + e2**2 + e3**2 + e4**2) * w).sum() / w.sum()
    # the same residuals written directly in the primitive (quasi-linear) form used by Msef
    ux, uy, vx, vy = _grad(u, x), _grad(u, y), _grad(v, x), _grad(v, y)
    px, py, rx, ry = _grad(p, x), _grad(p, y), _grad(rho, x), _grad(rho, y)
    a2 = e4_gamma_factor * p / rho
    f = [rho * ux + u * rx + rho * vy + v * ry, u * ux + v * uy + px / rho, u * vx + v * vy + py / rho,
         rho * a2 * (ux + vy) + u * px + v * py]
    alg = max((a - b).abs().max().item() for a, b in zip((e1, e2, e3, e4), f))
    return msef.item(), alg


@pytest.fixture(scope="module")
def polar_nn():
    ii, jj = np.meshgrid(np.arange(N_XI + 1), np.arange(N_ETA), indexing="ij")
    xx, yy = _polar(torch.tensor(ii, dtype=torch.float64), torch.tensor(jj, dtype=torch.float64))
    net = nf.NNfoil([2, 8, 8, 4], torch.device("cpu"), xx.numpy(), yy.numpy())
    net.model = _Manufactured()
    net.batch_ind = np.arange(net.XE.shape[0])
    return net


def test_e1_e4_equal_conservative_euler_residuals(polar_nn):
    """Msef (primitive e1-e4 + Jacobian chain rule) equals the conservative steady Euler residual on a smooth field."""
    with torch.enable_grad():
        got = polar_nn.Msef().item()
    ref, alg = _reference(polar_nn)
    assert got > 1e-3  # a non-trivial residual, so the comparison means something
    assert alg < 1e-9  # identities between the conservative and primitive forms hold to round-off
    assert abs(got - ref) / ref < 1e-3  # remaining difference is the finite-difference Jacobian (measured ~5e-5)


def test_e4_needs_the_gamma_in_a2(polar_nn):
    """Sensitivity: with a2 = p/rho (gamma dropped) the primitive e4 no longer matches the conservative energy."""
    _, alg = _reference(polar_nn, e4_gamma_factor=1.0)
    assert alg > 1e-2


# ---------------------------------------------------------------- finding 3: mesh generator


@pytest.fixture(scope="module")
def smoothing_rows():
    return {r["variant"]: r for r in check_mesh_smoothing.run()}


def test_finding3_upstream_reproduces_bundled_mesh(smoothing_rows):
    up = smoothing_rows["upstream"]
    assert up["max_dist_to_bundled"] < 1e-3  # the bundled mesh came from this code (measured 3.3e-4)
    assert up["jacobian_single_signed"]


def test_finding3_winslow_factor_is_better_converged(smoothing_rows):
    up, wi = smoothing_rows["upstream"], smoothing_rows["winslow"]
    assert wi["max_abs_winslow_residual"] < up["max_abs_winslow_residual"] / 10  # measured 2e-6 vs 2e-4
    assert wi["max_dist_to_bundled"] > 5e-3  # it is a different mesh from the bundled one (measured 0.011)
    assert wi["min_grid_angle_deg"] > up["min_grid_angle_deg"]
    assert wi["jacobian_single_signed"]


# ---------------------------------------------------------------- finding 10: Jacobian at the seam


def _wrong_sign_cells(nn_):
    s = torch.sign(nn_.J.reshape(-1))
    return int((s != s.median()).sum())


def test_finding10_periodic_flag_removes_wrong_sign_cells(mesh_50x25):
    xx, yy = mesh_50x25
    layers = [2, 8, 8, 4]
    off = nf.NNfoil(layers, torch.device("cpu"), xx, yy)
    on = nf.NNfoil(layers, torch.device("cpu"), xx, yy, periodic_jacobian=True)
    assert off.periodic_jacobian is False
    assert _wrong_sign_cells(off) >= 1
    assert _wrong_sign_cells(on) == 0
    assert on.J.shape == off.J.shape and on.Jac_inv.shape == off.Jac_inv.shape
    # away from the seam the two Jacobians agree
    assert torch.allclose(on.J.reshape(-1, xx.shape[1])[2:-2], off.J.reshape(-1, xx.shape[1])[2:-2], rtol=1e-4)


def test_default_jacobian_is_the_upstream_call(mesh_50x25):
    xx, yy = mesh_50x25
    net = nf.NNfoil([2, 8, 8, 4], torch.device("cpu"), xx, yy)
    nodes = torch.arange(-3, 4, 1, dtype=torch.long)
    jac, jac_inv = jacobian_trans(net.xx0.double(), net.yy.double(), nodes, boundary=(1, 1))
    assert torch.equal(net.J, torch.det(jac).reshape(-1, 1).float())
    assert torch.equal(net.Jac_inv, jac_inv.reshape(-1, 2, 2).float())


def test_finding10_all_bundled_meshes():
    rows = check_jacobian.run()
    assert len(rows) == 9
    for r in rows:
        assert 1 <= r["upstream_wrong_cells"] <= 2, r["mesh"]
        assert r["upstream_wrong_eta_cols"] == "0", r["mesh"]  # only the trailing-edge wall node
        assert r["periodic_wrong_cells"] == 0, r["mesh"]


# ---------------------------------------------------------------- weight initialisation (new in A9)


def test_normal_init_after_weight_norm_is_a_noop(mesh_50x25):
    """torch.nn.init.normal_(layer.weight) on a weight_norm layer is overwritten on the first forward pass:
    the effective weight is g * v / |v| (PyTorch's default Linear init), not N(0, 1)."""
    xx, yy = mesh_50x25
    torch.manual_seed(0)
    net = nf.NNfoil([2, 16, 16, 4], torch.device("cpu"), xx, yy)
    layer = net.model.layers[1]  # 16 -> 16
    assert layer.weight.std().item() > 0.7  # right after construction the plain attribute holds the N(0,1) draw
    with torch.no_grad():
        net.model(net.XE)
    expected = layer.weight_g * layer.weight_v / layer.weight_v.norm(dim=1, keepdim=True)
    assert torch.allclose(layer.weight, expected)
    assert layer.weight.std().item() < 0.3  # default init has std 1/sqrt(3*fan_in) = 0.14 for fan_in 16


# ---------------------------------------------------------------- Cp formula, seam and far-field points


class _ConstantPressure(torch.nn.Module):
    """Freestream velocity and density with p = p_far + dp (so Cp = dp / 0.5 by definition)."""

    def __init__(self, nn_, dp):
        super().__init__()
        self.c = torch.tensor([[nn_.far["u"].item(), nn_.far["v"].item(), nn_.far["p"].item() + dp, 1.0]])

    def forward(self, x):
        return self.c.expand(x.shape[0], 4) + 0 * x.sum(1, keepdim=True)


def test_cp_formula_is_pressure_coefficient(mesh_50x25):
    """Cp = (p - p_inf) / (0.5 rho_inf V_inf^2) with rho_inf = V_inf = 1: dp = 0.5 (= q_inf) is the stagnation value Cp = 1."""
    xx, yy = mesh_50x25
    net = nf.NNfoil([2, 8, 8, 4], torch.device("cpu"), xx, yy, Ma=0.4, alpha_deg=4.0)
    assert abs(net.far["p"].item() - 1 / (1.4 * 0.4**2)) < 1e-6
    assert abs(net.far["u"].item() ** 2 + net.far["v"].item() ** 2 - 1.0) < 1e-6  # |V_inf| = 1
    for dp, cp in ((0.5, 1.0), (-0.25, -0.5), (0.0, 0.0)):
        net.model = _ConstantPressure(net, dp)
        net.evaluate_wall()
        assert torch.allclose(net.wall["Cp_f"], torch.full_like(net.wall["Cp_f"], cp), atol=1e-5)
        assert torch.allclose(net.wall["Cp"], torch.full_like(net.wall["Cp"], cp), atol=1e-5)


def test_seam_and_far_field_collocation_points(mesh_50x25):
    xx, yy = mesh_50x25
    n_xi, n_eta = xx.shape
    net = nf.NNfoil([2, 8, 8, 4], torch.device("cpu"), xx, yy)
    left, right, far = net.bond_left.numpy(), net.bond_right.numpy(), net.far["XE"].numpy()
    eta = np.arange(n_eta)
    assert np.array_equal(left, np.c_[np.zeros(n_eta), eta])  # xi = 0 ...
    assert np.array_equal(right, np.c_[np.full(n_eta, n_xi - 1), eta])  # ... is paired with xi = N, same eta
    assert np.allclose(xx[0], xx[-1]) and np.allclose(yy[0], yy[-1])  # and those two rows are the same mesh nodes
    assert far.shape == (n_xi, 2) and (far[:, 1] == n_eta - 1).all()  # far field = the outermost eta row, all xi
