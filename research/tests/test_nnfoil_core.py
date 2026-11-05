import torch

import NNfoil_Euler2D as nf


def test_pick_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert nf.pick_device().type == "cpu"


def test_nnfoil_constructs_on_numpy2(mesh_50x25):
    xx, yy = mesh_50x25
    nn = nf.NNfoil([2, 16, 16, 4], torch.device("cpu"), xx, yy)
    assert nn.mx.shape[1] == nn.degree + 1
