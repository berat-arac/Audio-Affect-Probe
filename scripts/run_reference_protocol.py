from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print()
    print("RUN", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def train(kind: str, seed: int, device: str) -> None:
    run([
        sys.executable,
        "-m",
        "audio_affect_probe.train",
        "--config",
        f"configs/{kind}.yaml",
        "--device",
        device,
        "--seed",
        str(seed),
        "--zip-results",
        "--run-name",
        f"reference_{kind}_s{seed}",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the complete Audio Affect Probe reference protocol")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()

    print("Stage 1. Seed 42 architecture ablation.")
    for kind in ("baseline_mlp", "slow", "dual", "salience"):
        train(kind, 42, args.device)

    print("Stage 2. Fresh seed dual and salience comparison.")
    run([
        sys.executable,
        "scripts/run_gate_sweep.py",
        "--device",
        args.device,
        "--seeds",
        "1337",
        "2026",
        "31415",
    ])

    print("Stage 3. Separate head probe.")
    run([
        sys.executable,
        "scripts/run_head_probe.py",
        "--device",
        args.device,
        "--seed",
        "42",
    ])

    print("Stage 4. Fresh seed separate head confirmation.")
    run([
        sys.executable,
        "scripts/run_head_sweep.py",
        "--device",
        args.device,
        "--seeds",
        "1337",
        "2026",
        "31415",
    ])

    print()
    print("REFERENCE_PROTOCOL_COMPLETE")


if __name__ == "__main__":
    main()
