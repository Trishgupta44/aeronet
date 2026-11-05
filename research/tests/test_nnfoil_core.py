import torch

import NNfoil_Euler2D as nf


def test_pick_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert nf.pick_device().type == "cpu"
