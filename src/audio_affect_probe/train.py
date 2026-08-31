from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .data import CachedDEAMDataset, ChunkedDEAMDataset, collate_batch
from .metrics import loss_fn, regression_metrics
from .models import AffectModel, align_frame_features, align_frame_predictions
from .utils import choose_device, git_info, load_config, make_run_dir, package_run, save_json, seed_everything


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def summarize_gate_telemetry(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    gate = arrays["gate"]
    flux = arrays["flux"]
    energy_delta = arrays["energy_delta"]
    gate_delta = arrays["gate_delta"]
    dva = arrays["abs_delta_valence"]
    daa = arrays["abs_delta_arousal"]

    q05, q50, q95 = np.quantile(gate, [0.05, 0.50, 0.95])
    out = {
        "gate_mean": float(gate.mean()),
        "gate_std": float(gate.std()),
        "gate_p05": float(q05),
        "gate_p50": float(q50),
        "gate_p95": float(q95),
        "gate_dynamic_range_p95_p05": float(q95 - q05),
        "gate_flux_pearson": _corr(gate, flux),
        "gate_energy_delta_pearson": _corr(gate, energy_delta),
        "gate_abs_delta_valence_pearson": _corr(gate_delta, dva),
        "gate_abs_delta_arousal_pearson": _corr(gate_delta, daa),
    }

    if daa.size >= 10:
        threshold = float(np.quantile(daa, 0.90))
        hi = daa >= threshold
        lo = ~hi
        out["abs_delta_arousal_p90"] = threshold
        out["gate_mean_top10pct_arousal_change"] = float(gate_delta[hi].mean()) if hi.any() else 0.0
        out["gate_mean_other_arousal_change"] = float(gate_delta[lo].mean()) if lo.any() else 0.0
        denom = max(out["gate_mean_other_arousal_change"], 1e-12)
        out["gate_arousal_event_ratio"] = float(out["gate_mean_top10pct_arousal_change"] / denom)
    return out


def run_epoch(model, loader, device, cfg, optimizer=None, scaler=None, collect_telemetry: bool = False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    all_p, all_y = [], []
    gate_values = []
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    accum_steps = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))

    telemetry_parts: dict[str, list[np.ndarray]] = {
        "gate": [],
        "flux": [],
        "energy_delta": [],
        "gate_delta": [],
        "abs_delta_valence": [],
        "abs_delta_arousal": [],
    }

    if training:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(tqdm(loader, leave=False)):
            mel = batch["mel"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            frame_lengths = batch["frame_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                frame_pred, aux = model(mel)
                pred, mask = align_frame_predictions(frame_pred, frame_lengths, target_lengths)
                loss, _ = loss_fn(
                    pred, target, mask,
                    ccc_weight=float(cfg["loss"].get("ccc_weight", 1.0)),
                    mse_weight=float(cfg["loss"].get("mse_weight", 0.2)),
                )

            if training:
                backward_loss = loss / accum_steps
                is_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(loader))
                if scaler is not None and amp_enabled:
                    scaler.scale(backward_loss).backward()
                    if is_step:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 1.0)))
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    backward_loss.backward()
                    if is_step:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 1.0)))
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            total_batches += 1
            m = mask.detach().cpu().numpy().astype(bool)
            p = pred.detach().cpu().numpy()[m]
            y = target.detach().cpu().numpy()[m]
            all_p.append(p)
            all_y.append(y)

            if "gate" in aux:
                gate_values.append(aux["gate"].detach().mean().cpu().item())
                if collect_telemetry:
                    gate_t, _ = align_frame_features(aux["gate"], frame_lengths, target_lengths)
                    cue_t, _ = align_frame_features(aux["salience_cues"], frame_lengths, target_lengths)
                    for i in range(target.shape[0]):
                        t = int(target_lengths[i].item())
                        if t <= 0:
                            continue
                        g = gate_t[i, :t, 0].detach().float().cpu().numpy()
                        c = cue_t[i, :t].detach().float().cpu().numpy()
                        yy = target[i, :t].detach().float().cpu().numpy()
                        telemetry_parts["gate"].append(g)
                        telemetry_parts["flux"].append(c[:, 0])
                        telemetry_parts["energy_delta"].append(c[:, 1])
                        if t > 1:
                            dy = np.abs(yy[1:] - yy[:-1])
                            telemetry_parts["gate_delta"].append(g[1:])
                            telemetry_parts["abs_delta_valence"].append(dy[:, 0])
                            telemetry_parts["abs_delta_arousal"].append(dy[:, 1])

    pred_np = np.concatenate(all_p, axis=0)
    target_np = np.concatenate(all_y, axis=0)
    metrics = regression_metrics(pred_np, target_np)
    metrics["loss"] = total_loss / max(total_batches, 1)
    if gate_values:
        metrics["mean_gate"] = float(np.mean(gate_values))

    telemetry = None
    if collect_telemetry and telemetry_parts["gate"]:
        telemetry = {
            k: np.concatenate(v, axis=0).astype(np.float32)
            for k, v in telemetry_parts.items()
            if v
        }
    return metrics, pred_np, target_np, telemetry


def plot_history(df: pd.DataFrame, path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--zip-results", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=None, help="Override training seed without editing YAML")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)
    seed = int(cfg["training"].get("seed", 42))
    seed_everything(seed)
    device = choose_device(args.device)

    run_root = Path(cfg["output"].get("root", "results"))
    run_dir = make_run_dir(run_root, args.run_name or cfg["model"]["kind"])
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    save_json(git_info(), run_dir / "git.json")

    cache_dir = cfg["data"]["cache_dir"]
    chunk_targets = int(cfg["data"].get("chunk_targets", 0))
    if chunk_targets > 0:
        chunk_hop_targets = int(cfg["data"].get("chunk_hop_targets", chunk_targets))
        min_chunk_targets = int(cfg["data"].get("min_chunk_targets", max(4, chunk_targets // 3)))
        ds_kwargs = dict(
            chunk_targets=chunk_targets,
            chunk_hop_targets=chunk_hop_targets,
            min_chunk_targets=min_chunk_targets,
        )
        train_ds = ChunkedDEAMDataset(cache_dir, "train", **ds_kwargs)
        val_ds = ChunkedDEAMDataset(cache_dir, "val", **ds_kwargs)
        test_ds = ChunkedDEAMDataset(cache_dir, "test", **ds_kwargs)
        print(
            f"chunked_data targets/chunk={chunk_targets} hop={chunk_hop_targets} "
            f"chunks(train/val/test)={len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
        )
    else:
        train_ds = CachedDEAMDataset(cache_dir, "train")
        val_ds = CachedDEAMDataset(cache_dir, "val")
        test_ds = CachedDEAMDataset(cache_dir, "test")

    loader_kwargs = dict(
        batch_size=int(cfg["training"].get("batch_size", 16)),
        num_workers=int(cfg["training"].get("num_workers", 4)),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = AffectModel(
        kind=cfg["model"]["kind"],
        n_mels=int(cfg["model"].get("n_mels", 96)),
        hidden=int(cfg["model"].get("hidden", 128)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 3e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and bool(cfg["training"].get("amp", True))))

    patience = int(cfg["training"].get("patience", 8))
    best = -math.inf
    bad_epochs = 0
    history = []

    print(f"device={device} model={cfg['model']['kind']} seed={seed} params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, int(cfg["training"].get("epochs", 40)) + 1):
        train_m, _, _, _ = run_epoch(model, train_loader, device, cfg, optimizer, scaler)
        val_m, _, _, _ = run_epoch(model, val_loader, device, cfg)
        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_mean_ccc": train_m["mean_ccc"],
            "val_loss": val_m["loss"],
            "val_mean_ccc": val_m["mean_ccc"],
            "val_valence_ccc": val_m["valence_ccc"],
            "val_arousal_ccc": val_m["arousal_ccc"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(json.dumps(row))

        score = val_m["mean_ccc"]
        if score > best:
            best = score
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": cfg,
                "val_metrics": val_m,
            }, run_dir / "best.pt")
            save_json(val_m, run_dir / "best_val_metrics.json")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m, pred, target, telemetry = run_epoch(
        model, test_loader, device, cfg, collect_telemetry=(cfg["model"]["kind"] in {"salience", "affect_split"})
    )
    save_json(test_m, run_dir / "test_metrics.json")
    np.savez_compressed(run_dir / "test_predictions.npz", pred=pred, target=target)

    gate_summary = None
    if telemetry is not None:
        np.savez_compressed(run_dir / "test_gate_telemetry.npz", **telemetry)
        gate_summary = summarize_gate_telemetry(telemetry)
        save_json(gate_summary, run_dir / "gate_telemetry.json")

    hist_df = pd.DataFrame(history)
    plot_history(hist_df, run_dir / "training_curve.png")

    summary = {
        "device": str(device),
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": int(ckpt["epoch"]),
        "best_val": ckpt["val_metrics"],
        "test": test_m,
    }
    if gate_summary is not None:
        summary["gate_telemetry"] = gate_summary
    save_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2))
    print(f"RUN_DIR={run_dir}")

    if args.zip_results:
        zip_path = package_run(run_dir, cfg["output"].get("export_dir", "exports"))
        print(f"RESULT_ZIP={zip_path}")


if __name__ == "__main__":
    main()
