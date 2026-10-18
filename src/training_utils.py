from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from metrics import batch_sample_metrics, summarize_metric_rows


def evaluate_model(model, dataloader, device, *, max_batches: int | None = None):
    model.eval()
    rows = []
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for batch_idx, (seq, omics, hic) in enumerate(tqdm(dataloader, desc="evaluate", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            seq = seq.to(device, non_blocking=True)
            omics = omics.to(device, non_blocking=True)
            hic = hic.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(seq, omics)
            pred_np = pred.detach().cpu().numpy()
            hic_np = hic.detach().cpu().numpy()
            rows.extend(batch_sample_metrics(pred_np, hic_np))
            preds_all.append(pred_np)
            targets_all.append(hic_np)
    preds = np.concatenate(preds_all, axis=0) if preds_all else np.empty((0,))
    targets = np.concatenate(targets_all, axis=0) if targets_all else np.empty((0,))
    return summarize_metric_rows(rows), rows, preds, targets


def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    steps = 0
    for seq, omics, hic in tqdm(dataloader, desc="train", leave=False):
        seq = seq.to(device, non_blocking=True)
        omics = omics.to(device, non_blocking=True)
        hic = hic.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(seq, omics)
            loss = criterion(pred, hic)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
