from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def make_run_dir(root: str | Path, model_kind: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path(root) / f"{model_kind}_{stamp}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def git_info() -> dict:
    def run(args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return "unknown"
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status": run(["git", "status", "--short"]),
    }


def package_run(run_dir: str | Path, export_dir: str | Path = "exports") -> Path:
    run_dir = Path(run_dir)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    zip_path = export_dir / f"{run_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in run_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=f"{run_dir.name}/{p.relative_to(run_dir)}")
    return zip_path
