from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from audio_affect_probe.data import CachedDEAMDataset, ChunkedDEAMDataset, collate_batch
from audio_affect_probe.models import AffectModel
from audio_affect_probe.train import run_epoch, summarize_gate_telemetry
from audio_affect_probe.utils import choose_device, package_run, save_json, seed_everything


def main():
    ap = argparse.ArgumentParser(description="Evaluate an existing checkpoint and collect gate telemetry without retraining")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--output-root", default="results_eval")
    ap.add_argument("--export-dir", default="exports")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    seed = int(cfg["training"].get("seed", 42))
    seed_everything(seed)
    device = choose_device(args.device)

    cache_dir = cfg["data"]["cache_dir"]
    chunk_targets = int(cfg["data"].get("chunk_targets", 0))
    if chunk_targets > 0:
        ds = ChunkedDEAMDataset(
            cache_dir,
            "test",
            chunk_targets=chunk_targets,
            chunk_hop_targets=int(cfg["data"].get("chunk_hop_targets", chunk_targets)),
            min_chunk_targets=int(cfg["data"].get("min_chunk_targets", max(4, chunk_targets // 3))),
        )
    else:
        ds = CachedDEAMDataset(cache_dir, "test")

    loader = DataLoader(
        ds,
        shuffle=False,
        batch_size=int(cfg["training"].get("batch_size", 4)),
        num_workers=int(cfg["training"].get("num_workers", 2)),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = AffectModel(
        kind=cfg["model"]["kind"],
        n_mels=int(cfg["model"].get("n_mels", 96)),
        hidden=int(cfg["model"].get("hidden", 128)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    metrics, pred, target, telemetry = run_epoch(
        model,
        loader,
        device,
        cfg,
        collect_telemetry=(cfg["model"]["kind"] in {"salience", "affect_split"}),
    )

    out_dir = Path(args.output_root) / f"eval_{checkpoint.parent.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_dir / "test_metrics.json")
    np.savez_compressed(out_dir / "test_predictions.npz", pred=pred, target=target)

    summary = {
        "checkpoint": str(checkpoint),
        "model": cfg["model"]["kind"],
        "seed": seed,
        "best_epoch": int(ckpt.get("epoch", -1)),
        "test": metrics,
    }
    if telemetry is not None:
        gate_summary = summarize_gate_telemetry(telemetry)
        save_json(gate_summary, out_dir / "gate_telemetry.json")
        np.savez_compressed(out_dir / "test_gate_telemetry.npz", **telemetry)
        summary["gate_telemetry"] = gate_summary

    save_json(summary, out_dir / "summary.json")
    zip_path = package_run(out_dir, args.export_dir)
    print(json.dumps(summary, indent=2))
    print(f"EVAL_ZIP={zip_path}")


if __name__ == "__main__":
    main()
