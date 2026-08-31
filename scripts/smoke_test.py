from __future__ import annotations

import tempfile
from pathlib import Path
import json
import torch

from audio_affect_probe.data import CachedDEAMDataset, collate_batch
from audio_affect_probe.metrics import loss_fn
from audio_affect_probe.models import AffectModel, align_frame_predictions


def main():
    torch.manual_seed(0)
    for kind in ["baseline_mlp", "slow", "dual", "salience", "dual_split", "affect_split"]:
        model = AffectModel(kind=kind, n_mels=96, hidden=32, dropout=0.0)
        mel = torch.rand(3, 180, 96)
        frame_lengths = torch.tensor([180, 150, 120])
        target_lengths = torch.tensor([61, 51, 41])
        target = torch.rand(3, 61, 2) * 2 - 1
        frame_pred, aux = model(mel)
        pred, mask = align_frame_predictions(frame_pred, frame_lengths, target_lengths)
        loss, _ = loss_fn(pred, target, mask, 1.0, 0.2)
        loss.backward()
        assert torch.isfinite(loss)
        print(kind, "ok", "loss=", round(float(loss.detach()), 4), "params=", sum(p.numel() for p in model.parameters()))
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
