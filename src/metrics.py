from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr


def safe_corr(x, y, method: str = "pearson") -> float:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.nan_to_num(np.asarray(y, dtype=np.float64).reshape(-1))
    if x.size < 2 or y.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    corr_func = pearsonr if method == "pearson" else spearmanr
    value, _ = corr_func(y, x)
    return float(value)


def batch_sample_metrics(preds: np.ndarray, targets: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for pred, target in zip(preds, targets):
        rows.append(
            {
                "pcc": safe_corr(pred, target, "pearson"),
                "spearman": safe_corr(pred, target, "spearman"),
                "mae": float(np.mean(np.abs(pred - target))),
                "mse": float(np.mean((pred - target) ** 2)),
            }
        )
    return rows


def summarize_metric_rows(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    summary = {"n_samples": float(len(rows))}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(np.nanmean(values))
        summary[f"median_{key}"] = float(np.nanmedian(values))
    return summary


def distance_pcc(preds: np.ndarray, targets: np.ndarray, max_dist: int | None = None) -> list[float]:
    if max_dist is None:
        max_dist = preds.shape[-1]
    values: list[float] = []
    for dist in range(1, max_dist):
        pred_diag = np.concatenate([np.diagonal(sample, offset=dist) for sample in preds])
        target_diag = np.concatenate([np.diagonal(sample, offset=dist) for sample in targets])
        values.append(safe_corr(pred_diag, target_diag, "pearson"))
    return values


def insulation_score(matrix: np.ndarray, windowsize: int = 500000, resolution: int = 10000) -> np.ndarray:
    window_bins = int(windowsize / resolution)
    score = np.ones((matrix.shape[0]), dtype=np.float32)
    for i in range(matrix.shape[0]):
        with np.errstate(divide="ignore", invalid="ignore"):
            diamond_sum = np.sum(matrix[max(0, i - window_bins) : i, i + 1 : min(matrix.shape[0], i + window_bins + 1)])
            square_sum = np.sum(
                matrix[
                    max(0, i - window_bins) : min(matrix.shape[0], i + window_bins + 1),
                    max(0, i - window_bins) : min(matrix.shape[0], i + window_bins + 1),
                ]
            )
            value = diamond_sum / square_sum if square_sum else math.nan
            score[i] = value if np.isfinite(value) else 1.0
    return score
