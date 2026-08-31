from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from audio_affect_probe.audio import load_audio_segment, log_mel

SAMPLE_RE = re.compile(r"sample_(\d+)ms")
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def norm_id(x) -> str:
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except Exception:
        return s.lstrip("0") or "0"


def build_audio_index(root: Path) -> dict[str, Path]:
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out[norm_id(p.stem)] = p
    return out


def find_id_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).lower() in {"song_id", "songid", "id"}:
            return c
    return df.columns[0]


def sample_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if SAMPLE_RE.fullmatch(str(c))]
    return sorted(cols, key=lambda c: int(SAMPLE_RE.fullmatch(str(c)).group(1)))


def main():
    ap = argparse.ArgumentParser(description="Precompute DEAM log mel features aligned to dynamic valence and arousal annotations.")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--arousal-csv", required=True)
    ap.add_argument("--valence-csv", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=96)
    ap.add_argument("--hop-length", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0 = all songs; useful for a quick test")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a_df = pd.read_csv(args.arousal_csv)
    v_df = pd.read_csv(args.valence_csv)
    a_id, v_id = find_id_col(a_df), find_id_col(v_df)
    a_df["__id"] = a_df[a_id].map(norm_id)
    v_df["__id"] = v_df[v_id].map(norm_id)
    a_df = a_df.set_index("__id", drop=False)
    v_df = v_df.set_index("__id", drop=False)

    common_cols = sorted(
        set(sample_columns(a_df)).intersection(sample_columns(v_df)),
        key=lambda c: int(SAMPLE_RE.fullmatch(str(c)).group(1)),
    )
    if not common_cols:
        raise RuntimeError("No common sample_<N>ms columns found in annotation CSVs")

    audio_index = build_audio_index(Path(args.audio_dir))
    ids = sorted(set(a_df.index).intersection(v_df.index).intersection(audio_index.keys()))
    if args.limit > 0:
        ids = ids[: args.limit]
    if not ids:
        raise RuntimeError("No matching song IDs between audio and annotation files")

    rng = random.Random(args.seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1)) if n >= 3 else 0
    split_map = {}
    for i, sid in enumerate(shuffled):
        split_map[sid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    # Ensure a test split for very small smoke preparations.
    if n >= 2 and not any(v == "test" for v in split_map.values()):
        split_map[shuffled[-1]] = "test"
    if n >= 3 and not any(v == "val" for v in split_map.values()):
        split_map[shuffled[-2]] = "val"

    items = []
    skipped = []
    for sid in tqdm(ids):
        ar = pd.to_numeric(a_df.loc[sid, common_cols], errors="coerce").to_numpy(dtype=np.float32)
        va = pd.to_numeric(v_df.loc[sid, common_cols], errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(ar) & np.isfinite(va)
        if finite.sum() < 4:
            skipped.append({"song_id": sid, "reason": "too_few_finite_targets"})
            continue

        cols = [c for c, ok in zip(common_cols, finite) if ok]
        ar = ar[finite]
        va = va[finite]
        timestamps_ms = np.array([int(SAMPLE_RE.fullmatch(str(c)).group(1)) for c in cols], dtype=np.int64)
        # Keep only the largest evenly spaced contiguous block so interpolation remains meaningful.
        if len(timestamps_ms) > 1:
            step = int(np.median(np.diff(timestamps_ms)))
            breaks = np.where(np.diff(timestamps_ms) != step)[0]
            blocks = np.split(np.arange(len(timestamps_ms)), breaks + 1)
            block = max(blocks, key=len)
            timestamps_ms, ar, va = timestamps_ms[block], ar[block], va[block]

        start_s = float(timestamps_ms[0]) / 1000.0
        end_s = float(timestamps_ms[-1]) / 1000.0 + 0.5
        try:
            y = load_audio_segment(audio_index[sid], args.sr, start_s, end_s)
            mel = log_mel(y, sr=args.sr, hop_length=args.hop_length, n_mels=args.n_mels)
        except Exception as e:
            skipped.append({"song_id": sid, "reason": f"audio_error: {e}"})
            continue

        target = torch.from_numpy(np.stack([va, ar], axis=-1).astype(np.float32))
        cache_file = f"song_{sid}.pt"
        torch.save({
            "song_id": sid,
            "mel": mel,
            "target": target,
            "timestamps_ms": torch.from_numpy(timestamps_ms),
            "audio_path": str(audio_index[sid]),
        }, out / cache_file)
        items.append({
            "song_id": sid,
            "split": split_map[sid],
            "cache_file": cache_file,
            "frames": int(mel.shape[0]),
            "targets": int(target.shape[0]),
        })

    counts = {s: sum(x["split"] == s for x in items) for s in ["train", "val", "test"]}
    manifest = {
        "format_version": 1,
        "sr": args.sr,
        "n_mels": args.n_mels,
        "hop_length": args.hop_length,
        "items": items,
        "split_counts": counts,
        "skipped": skipped,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"prepared": len(items), "splits": counts, "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    main()
