from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset


class CachedDEAMDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}. Run scripts/prepare_deam.py first.")
        manifest = json.loads(manifest_path.read_text())
        self.items = [x for x in manifest["items"] if x["split"] == split]
        if not self.items:
            raise RuntimeError(f"No items for split={split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self.items[idx]
        return torch.load(self.cache_dir / meta["cache_file"], map_location="cpu", weights_only=False)


class ChunkedDEAMDataset(Dataset):
    """View the song level cache as fixed size temporal chunks.

    Splits remain song level because the manifest is split before chunking. This avoids
    train and validation leakage while keeping recurrent activations bounded on small GPUs.
    chunk_targets is expressed in DEAM target frames (normally one target every 0.5 s).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        chunk_targets: int = 60,
        chunk_hop_targets: int | None = None,
        min_chunk_targets: int = 20,
    ):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}. Run scripts/prepare_deam.py first.")
        manifest = json.loads(manifest_path.read_text())
        songs = [x for x in manifest["items"] if x["split"] == split]
        if not songs:
            raise RuntimeError(f"No items for split={split!r} in {manifest_path}")

        self.chunk_targets = int(chunk_targets)
        self.chunk_hop_targets = int(chunk_hop_targets or chunk_targets)
        self.min_chunk_targets = int(min_chunk_targets)
        if self.chunk_targets <= 0 or self.chunk_hop_targets <= 0:
            raise ValueError("chunk_targets and chunk_hop_targets must be positive")

        self.items: list[dict[str, Any]] = []
        for song in songs:
            total_t = int(song["targets"])
            if total_t < self.min_chunk_targets:
                continue
            start = 0
            while start < total_t:
                end = min(start + self.chunk_targets, total_t)
                if end - start >= self.min_chunk_targets:
                    self.items.append({"song": song, "t0": start, "t1": end})
                if end == total_t:
                    break
                start += self.chunk_hop_targets

        if not self.items:
            raise RuntimeError(f"No chunks for split={split!r}; lower min_chunk_targets")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self.items[idx]
        song = meta["song"]
        obj = torch.load(self.cache_dir / song["cache_file"], map_location="cpu", weights_only=False)
        mel = obj["mel"]
        target = obj["target"]
        total_f = int(mel.shape[0])
        total_t = int(target.shape[0])
        t0, t1 = int(meta["t0"]), int(meta["t1"])

        # Map target boundaries proportionally to mel frame boundaries. This handles
        # the one frame differences introduced by centered STFT padding safely.
        f0 = int(round(t0 * total_f / total_t))
        f1 = int(round(t1 * total_f / total_t))
        f0 = max(0, min(f0, total_f - 1))
        f1 = max(f0 + 1, min(f1, total_f))

        timestamps = obj.get("timestamps_ms")
        return {
            "song_id": f"{obj['song_id']}:{t0}-{t1}",
            "mel": mel[f0:f1],
            "target": target[t0:t1],
            "timestamps_ms": timestamps[t0:t1] if timestamps is not None else None,
            "audio_path": obj.get("audio_path", ""),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    bsz = len(batch)
    max_frames = max(x["mel"].shape[0] for x in batch)
    max_targets = max(x["target"].shape[0] for x in batch)
    n_mels = batch[0]["mel"].shape[1]

    mel = torch.zeros(bsz, max_frames, n_mels, dtype=torch.float32)
    target = torch.zeros(bsz, max_targets, 2, dtype=torch.float32)
    target_mask = torch.zeros(bsz, max_targets, dtype=torch.bool)
    frame_lengths = torch.zeros(bsz, dtype=torch.long)
    target_lengths = torch.zeros(bsz, dtype=torch.long)
    song_ids: list[str] = []

    for i, item in enumerate(batch):
        f = item["mel"].shape[0]
        t = item["target"].shape[0]
        mel[i, :f] = item["mel"]
        target[i, :t] = item["target"]
        target_mask[i, :t] = True
        frame_lengths[i] = f
        target_lengths[i] = t
        song_ids.append(str(item["song_id"]))

    return {
        "mel": mel,
        "target": target,
        "target_mask": target_mask,
        "frame_lengths": frame_lengths,
        "target_lengths": target_lengths,
        "song_ids": song_ids,
    }
