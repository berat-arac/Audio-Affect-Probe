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

import pandas as pd


def run_one(root: Path, seed: int, device: str):
    run_name = f"head_dual_split_s{seed}"
    cmd = [
        sys.executable, "-m", "audio_affect_probe.train",
        "--config", "configs/dual_split.yaml",
        "--device", device,
        "--seed", str(seed),
        "--zip-results",
        "--run-name", run_name,
    ]
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"\n=== DUAL_SPLIT seed {seed} ===", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd, cwd=root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    run_dir = None
    result_zip = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        s = line.strip()
        if s.startswith("RUN_DIR="):
            run_dir = (root / s.split("=", 1)[1]).resolve()
        elif s.startswith("RESULT_ZIP="):
            result_zip = (root / s.split("=", 1)[1]).resolve()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"dual_split seed={seed} failed with exit code {code}")
    if run_dir is None or result_zip is None:
        raise RuntimeError(f"Could not locate outputs for seed={seed}")

    summary = json.loads((run_dir / "summary.json").read_text())
    t = summary["test"]
    row = {
        "model": "dual_split",
        "seed": seed,
        "best_epoch": summary["best_epoch"],
        "parameters": summary["parameters"],
        "valence_ccc": t["valence_ccc"],
        "arousal_ccc": t["arousal_ccc"],
        "mean_ccc": t["mean_ccc"],
        "valence_rmse": t["valence_rmse"],
        "arousal_rmse": t["arousal_rmse"],
        "zip": result_zip.name,
    }
    return row, result_zip


def main():
    ap = argparse.ArgumentParser(description="Fresh seed confirmation sweep for dual_split")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 2026, 31415])
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = root / "exports" / f"head_dual_split_sweep_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    zips = []
    for seed in args.seeds:
        row, zp = run_one(root, seed, args.device)
        rows.append(row)
        zips.append(zp)
        pd.DataFrame(rows).to_csv(bundle_dir / "sweep_summary_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(bundle_dir / "sweep_summary.csv", index=False)
    (bundle_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2))

    aggregate = {}
    for metric in ["valence_ccc", "arousal_ccc", "mean_ccc", "valence_rmse", "arousal_rmse"]:
        vals = df[metric].astype(float)
        aggregate[metric] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        }
    (bundle_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))

    for zp in zips:
        shutil.copy2(zp, bundle_dir / zp.name)

    (bundle_dir / "README.txt").write_text(
        "Audio Affect Probe dual_split confirmation sweep\n"
        f"Seeds: {args.seeds}\n"
        "Same DEAM song split; only training/model randomness changes.\n"
    )

    bundle_zip = root / "exports" / f"head_dual_split_sweep_{stamp}.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in bundle_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=f"{bundle_dir.name}/{p.relative_to(bundle_dir)}")

    print("\n=== DUAL_SPLIT SWEEP COMPLETE ===")
    print(df[["seed", "valence_ccc", "arousal_ccc", "mean_ccc"]].to_string(index=False))
    print(f"SWEEP_BUNDLE={bundle_zip}")


if __name__ == "__main__":
    main()
