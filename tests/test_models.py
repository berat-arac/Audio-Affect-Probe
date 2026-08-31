import torch
from audio_affect_probe.models import AffectModel, align_frame_predictions


def test_model_shapes():
    x = torch.rand(2, 100, 96)
    for kind in ["baseline_mlp", "slow", "dual", "salience", "dual_split", "affect_split"]:
        model = AffectModel(kind, n_mels=96, hidden=32, dropout=0.0)
        frame, aux = model(x)
        assert frame.shape == (2, 100, 2)
        aligned, mask = align_frame_predictions(frame, torch.tensor([100, 80]), torch.tensor([61, 41]))
        assert aligned.shape == (2, 61, 2)
        assert mask.sum().item() == 102


def test_affect_split_has_gate():
    x = torch.rand(2, 100, 96)
    model = AffectModel("affect_split", n_mels=96, hidden=32, dropout=0.0)
    frame, aux = model(x)
    assert frame.shape == (2, 100, 2)
    assert "gate" in aux
    assert aux["gate"].shape == (2, 100, 1)
