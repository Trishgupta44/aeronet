import torch

import NNfoil_Euler2D as nf


def test_pick_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert nf.pick_device().type == "cpu"


def test_nnfoil_constructs_on_numpy2(mesh_50x25):
    xx, yy = mesh_50x25
    nn = nf.NNfoil([2, 16, 16, 4], torch.device("cpu"), xx, yy)
    assert nn.mx.shape[1] == nn.degree + 1


def test_checkpoint_roundtrip(tmp_path, mesh_50x25):
    xx, yy = mesh_50x25
    ckpt = tmp_path / "model.pt"
    nn = nf.NNfoil([2, 16, 16, 4], torch.device("cpu"), xx, yy, ckpt_path=str(ckpt))
    with torch.no_grad():
        saved = nn.model(nn.XE).clone()
    nn._save_ckpt()
    assert ckpt.exists()
    with torch.no_grad():
        for prm in nn.model.parameters():
            prm.add_(torch.randn_like(prm))
        assert not torch.equal(nn.model(nn.XE), saved)
    nn._load_ckpt()
    with torch.no_grad():
        assert torch.equal(nn.model(nn.XE), saved)


def _scaled_net_outputs(mesh, positivity):
    xx, yy = mesh
    torch.manual_seed(0)
    nn = nf.NNfoil([2, 16, 16, 4], torch.device("cpu"), xx, yy, positivity=positivity)
    with torch.no_grad():
        for name, prm in nn.model.named_parameters():
            if name.endswith("weight_g"):
                prm.mul_(50.0)
        return nn.model(nn.XE)


def test_positivity_keeps_p_rho_above_floor(mesh_50x25):
    out = _scaled_net_outputs(mesh_50x25, positivity=True)
    assert out[:, 2].min().item() >= 1e-3 - 1e-9
    assert out[:, 3].min().item() >= 1e-3 - 1e-9


def test_positivity_off_is_raw(mesh_50x25):
    out = _scaled_net_outputs(mesh_50x25, positivity=False)
    assert out[:, 2].min().item() < 0


def test_flow_condition_arguments(mesh_50x25):
    import math

    xx, yy = mesh_50x25
    nn = nf.NNfoil([2, 16, 16, 4], torch.device("cpu"), xx, yy, Ma=0.3, alpha_deg=2.0)
    assert abs(nn.far["u"].item() - math.cos(math.radians(2.0))) < 1e-6
    assert abs(nn.far["v"].item() - math.sin(math.radians(2.0))) < 1e-6
    assert abs(nn.far["p"].item() - 1 / (1.4 * 0.09)) < 1e-4


def test_evaluate_wall_returns_finite_floats(mesh_50x25):
    import math

    xx, yy = mesh_50x25
    nn = nf.NNfoil([2, 16, 16, 4], torch.device("cpu"), xx, yy)
    cl, cd = nn.evaluate_wall()
    assert isinstance(cl, float) and isinstance(cd, float)
    assert math.isfinite(cl) and math.isfinite(cd)
    assert nn.wall["Cp_f"].shape == (xx.shape[0] - 1, 1)
