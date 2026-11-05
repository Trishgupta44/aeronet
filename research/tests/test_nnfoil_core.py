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
