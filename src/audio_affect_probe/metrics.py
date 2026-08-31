from __future__ import annotations

import math
import numpy as np
import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).expand_as(pred)
    diff = (pred - target)[m]
    return (diff * diff).mean()


def masked_ccc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scores = []
    for d in range(pred.shape[-1]):
        p = pred[..., d][mask]
        y = target[..., d][mask]
        mp, my = p.mean(), y.mean()
        vp = ((p - mp) ** 2).mean()
        vy = ((y - my) ** 2).mean()
        cov = ((p - mp) * (y - my)).mean()
        ccc = 2.0 * cov / (vp + vy + (mp - my) ** 2 + eps)
        scores.append(ccc)
    return torch.stack(scores).mean()


def loss_fn(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, ccc_weight: float, mse_weight: float):
    ccc = masked_ccc(pred, target, mask)
    mse = masked_mse(pred, target, mask)
    loss = ccc_weight * (1.0 - ccc) + mse_weight * mse
    return loss, {"ccc": ccc.detach(), "mse": mse.detach()}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    return float((2.0 * cov) / (vx + vy + (mx - my) ** 2 + 1e-12))


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    names = ["valence", "arousal"]
    for d, name in enumerate(names):
        p, y = pred[:, d], target[:, d]
        out[f"{name}_ccc"] = _ccc(p, y)
        out[f"{name}_pearson"] = _pearson(p, y)
        out[f"{name}_rmse"] = float(math.sqrt(np.mean((p - y) ** 2)))
    out["mean_ccc"] = 0.5 * (out["valence_ccc"] + out["arousal_ccc"])
    out["mean_rmse"] = 0.5 * (out["valence_rmse"] + out["arousal_rmse"])
    return out
