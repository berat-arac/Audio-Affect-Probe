from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def run_one(root: Path, model: str, seed: int, device: str) -> tuple[dict, Path]:
    cmd = [
        sys.executable,
        "-m",
        "audio_affect_probe.train",
        "--config",
        f"configs/{model}.yaml",
        "--device",
        device,
        "--zip-results",
        "--seed",
        str(seed),
        "--run-name",
        f"gate_{model}_s{seed}",
    ]
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("\n=== RUN", model, "seed", seed, "===", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    run_dir = None
    result_zip = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        line = line.strip()
        if line.startswith("RUN_DIR="):
            run_dir = (root / line.split("=", 1)[1]).resolve()
        elif line.startswith("RESULT_ZIP="):
            result_zip = (root / line.split("=", 1)[1]).resolve()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Training failed for model={model} seed={seed} with exit code {code}")
    if run_dir is None or result_zip is None:
        raise RuntimeError(f"Could not locate outputs for model={model} seed={seed}")

    summary = json.loads((run_dir / "summary.json").read_text())
    row = {
        "model": model,
        "seed": seed,
        "best_epoch": summary["best_epoch"],
        "parameters": summary["parameters"],
        "valence_ccc": summary["test"]["valence_ccc"],
        "arousal_ccc": summary["test"]["arousal_ccc"],
        "mean_ccc": summary["test"]["mean_ccc"],
        "valence_rmse": summary["test"]["valence_rmse"],
        "arousal_rmse": summary["test"]["arousal_rmse"],
        "zip": result_zip.name,
    }
    gate = summary.get("gate_telemetry")
    if gate:
        for k, v in gate.items():
            row[f"gate__{k}"] = v
    return row, result_zip


def main():
    ap = argparse.ArgumentParser(description="Paired fresh seed comparison for Dual and Salience")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 2026, 31415])
    ap.add_argument("--models", nargs="+", default=["dual", "salience"], choices=["dual", "salience"])
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = root / "exports" / f"gate_sweep_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict] = []
    zips: list[Path] = []
    for seed in args.seeds:
        for model in args.models:
            row, result_zip = run_one(root, model, seed, args.device)
            rows.append(row)
            zips.append(result_zip)
            pd.DataFrame(rows).to_csv(bundle_dir / "sweep_summary_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(bundle_dir / "sweep_summary.csv", index=False)
    (bundle_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2))

    aggregate = {}
    for model, g in df.groupby("model"):
        aggregate[model] = {}
        for metric in ["valence_ccc", "arousal_ccc", "mean_ccc", "valence_rmse", "arousal_rmse"]:
            vals = g[metric].to_numpy(dtype=float)
            aggregate[model][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
            }

    paired = []
    if {"dual", "salience"}.issubset(set(df["model"])):
        d = df[df.model == "dual"].set_index("seed")
        s = df[df.model == "salience"].set_index("seed")
        for seed in sorted(set(d.index).intersection(s.index)):
            paired.append({
                "seed": int(seed),
                "delta_arousal_ccc_salience_minus_dual": float(s.loc[seed, "arousal_ccc"] - d.loc[seed, "arousal_ccc"]),
                "delta_valence_ccc_salience_minus_dual": float(s.loc[seed, "valence_ccc"] - d.loc[seed, "valence_ccc"]),
                "delta_mean_ccc_salience_minus_dual": float(s.loc[seed, "mean_ccc"] - d.loc[seed, "mean_ccc"]),
            })
        if paired:
            pdf = pd.DataFrame(paired)
            pdf.to_csv(bundle_dir / "paired_deltas.csv", index=False)
            aggregate["paired"] = {
                "n": len(paired),
                "mean_delta_arousal_ccc": float(pdf["delta_arousal_ccc_salience_minus_dual"].mean()),
                "mean_delta_valence_ccc": float(pdf["delta_valence_ccc_salience_minus_dual"].mean()),
                "mean_delta_mean_ccc": float(pdf["delta_mean_ccc_salience_minus_dual"].mean()),
                "salience_arousal_wins": int((pdf["delta_arousal_ccc_salience_minus_dual"] > 0).sum()),
            }
    (bundle_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))

    for zp in zips:
        shutil.copy2(zp, bundle_dir / zp.name)

    readme = f"""Audio Affect Probe gate fresh seed sweep\n\nSeeds: {args.seeds}\nModels: {args.models}\n\nThis bundle contains full per run ZIPs, sweep_summary.csv/json, paired_deltas.csv, and aggregate.json.\nThe DEAM song split is unchanged; --seed changes training/model randomness only.\n"""
    (bundle_dir / "README.txt").write_text(readme)

    bundle_zip = root / "exports" / f"gate_sweep_{stamp}.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in bundle_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=f"{bundle_dir.name}/{p.relative_to(bundle_dir)}")

    print("\n=== SWEEP COMPLETE ===")
    print(df[["model", "seed", "valence_ccc", "arousal_ccc", "mean_ccc"]].to_string(index=False))
    if paired:
        print("\nPaired deltas (Salience minus Dual):")
        print(pd.DataFrame(paired).to_string(index=False))
    print(f"SWEEP_BUNDLE={bundle_zip}")


if __name__ == "__main__":
    main()
