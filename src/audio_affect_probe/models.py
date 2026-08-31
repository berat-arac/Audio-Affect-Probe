from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SharedProjection(nn.Module):
    def __init__(self, n_mels: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_mels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastPath(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        layers = []
        for dilation in (1, 2, 4):
            layers.extend([
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2 * dilation, dilation=dilation, groups=hidden),
                nn.Conv1d(hidden, hidden, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class SlowPath(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.norm(y)


def salience_cues(mel: torch.Tensor) -> torch.Tensor:
    # mel: [B,T,M]. Two transparent cues: relative energy and spectral flux.
    energy = mel.mean(dim=-1, keepdim=True)
    prev = F.pad(mel[:, :-1], (0, 0, 1, 0))
    flux = (mel - prev).abs().mean(dim=-1, keepdim=True)
    energy_delta = F.pad(energy[:, 1:] - energy[:, :-1], (0, 0, 1, 0)).abs()
    return torch.cat([flux, energy_delta], dim=-1)


class AffectModel(nn.Module):
    def __init__(self, kind: str, n_mels: int = 96, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.kind = kind
        self.proj = SharedProjection(n_mels, hidden, dropout)
        self.fast = FastPath(hidden, dropout) if kind in {"dual", "salience", "dual_split", "affect_split"} else None
        self.slow = SlowPath(hidden, dropout) if kind in {"slow", "dual", "salience", "dual_split", "affect_split"} else None

        if kind == "baseline_mlp":
            fused_dim = hidden
        elif kind == "slow":
            fused_dim = hidden
        elif kind == "dual":
            self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
            fused_dim = hidden
        elif kind == "salience":
            self.gate = nn.Sequential(
                nn.Linear(hidden * 2 + 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )
            self.fuse_norm = nn.LayerNorm(hidden)
            fused_dim = hidden
        elif kind == "dual_split":
            # Same representation as Dual, but separate one dimensional heads.
            # This isolates ordinary multi task/head interference from fusion effects.
            self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
            fused_dim = hidden
        elif kind == "affect_split":
            # Dimension specific fusion:
            # valence gets the unrestricted dual path representation,
            # arousal gets a salience gated fast residual over the slow state.
            self.valence_fuse = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden * 2 + 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )
            self.arousal_norm = nn.LayerNorm(hidden)
            fused_dim = hidden
        else:
            raise ValueError(f"Unknown model kind: {kind}")

        if kind in {"dual_split", "affect_split"}:
            def scalar_head():
                return nn.Sequential(
                    nn.Linear(fused_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                    nn.Tanh(),
                )
            self.valence_head = scalar_head()
            self.arousal_head = scalar_head()
            self.head = None
        else:
            self.head = nn.Sequential(
                nn.Linear(fused_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
                nn.Tanh(),
            )

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.proj(mel)
        aux: dict[str, torch.Tensor] = {}

        if self.kind == "baseline_mlp":
            z = x
        elif self.kind == "slow":
            z = self.slow(x)
        elif self.kind == "dual":
            f = self.fast(x)
            s = self.slow(x)
            z = self.fuse(torch.cat([f, s], dim=-1))
        elif self.kind == "salience":
            f = self.fast(x)
            s = self.slow(x)
            cues = salience_cues(mel)
            gate = self.gate(torch.cat([f, s, cues], dim=-1))
            z = self.fuse_norm(s + gate * f)
            aux["gate"] = gate
            aux["salience_cues"] = cues
        elif self.kind == "dual_split":
            f = self.fast(x)
            s = self.slow(x)
            z = self.fuse(torch.cat([f, s], dim=-1))
            valence = self.valence_head(z)
            arousal = self.arousal_head(z)
            return torch.cat([valence, arousal], dim=-1), aux
        else:  # affect_split
            f = self.fast(x)
            s = self.slow(x)
            cues = salience_cues(mel)
            gate = self.gate(torch.cat([f, s, cues], dim=-1))
            z_valence = self.valence_fuse(torch.cat([f, s], dim=-1))
            z_arousal = self.arousal_norm(s + gate * f)
            aux["gate"] = gate
            aux["salience_cues"] = cues
            aux["valence_state"] = z_valence
            aux["arousal_state"] = z_arousal
            valence = self.valence_head(z_valence)
            arousal = self.arousal_head(z_arousal)
            return torch.cat([valence, arousal], dim=-1), aux

        return self.head(z), aux


def align_frame_features(
    frame_feat: torch.Tensor,
    frame_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly align arbitrary frame level features [B,F,D] to target rate."""
    bsz = frame_feat.shape[0]
    dims = frame_feat.shape[-1]
    max_target = int(target_lengths.max().item())
    out = frame_feat.new_zeros((bsz, max_target, dims))
    mask = torch.zeros((bsz, max_target), dtype=torch.bool, device=frame_feat.device)
    for i in range(bsz):
        f = int(frame_lengths[i].item())
        t = int(target_lengths[i].item())
        seq = frame_feat[i, :f].transpose(0, 1).unsqueeze(0)  # [1,D,F]
        aligned = F.interpolate(seq, size=t, mode="linear", align_corners=False)
        out[i, :t] = aligned.squeeze(0).transpose(0, 1)
        mask[i, :t] = True
    return out, mask


def align_frame_predictions(
    frame_pred: torch.Tensor,
    frame_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return align_frame_features(frame_pred, frame_lengths, target_lengths)
