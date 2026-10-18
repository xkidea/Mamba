#!/usr/bin/env python
"""Train the recovered GM12878 ``CNNMambaHybridModelLegacy`` model.

This bundle-specific entry point intentionally supports only the Legacy model.
It follows the recovered ``train.ipynb`` protocol: no augmentation, log2 Hi-C,
a zeroed Hi-C diagonal and Gaussian distance-weighted MSE.  By default chr7 is
the only holdout chromosome, so chr22 remains in the training set.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class NotebookProfile:
    """Legacy defaults recovered from the closest training notebook."""

    model_class: str
    batch_size: int
    default_excluded_chromosomes: tuple[str, ...]
    best_model_filename: str
    provenance: str


LEGACY_PROFILE = NotebookProfile(
    model_class="CNNMambaHybridModelLegacy",
    batch_size=18,
    default_excluded_chromosomes=(),
    best_model_filename="best_model.pth",
    provenance=(
        "train.ipynb, catalogued in "
        "main_figures/figure_06_loss_weights/README.md"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain the checkpoint-compatible CNNMambaHybridModelLegacy model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        choices=("legacy",),
        default="legacy",
        help="Compatibility option; this dedicated script supports only Legacy.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "processed" / "preprocessed_data_omic_512"),
        help="Directory containing GM12878 pickle samples.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. By default, results/training/cnn_mamba_legacy is used.",
    )

    # Recovered notebook split and optimization settings.
    parser.add_argument("--holdout-chromosome", default="chr7")
    parser.add_argument(
        "--exclude-chromosomes",
        default=None,
        help=(
            "Comma-separated chromosomes excluded from training in addition to the holdout. "
            "The Legacy default is none."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to 18, following the closest Legacy notebook.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-parallel", action="store_true")

    # Architecture values used by the recovered Legacy checkpoint.
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--fusion-out-dim", type=int, default=128)
    parser.add_argument("--num-cnn-blocks", type=int, default=3)
    parser.add_argument("--num-mamba-layers", type=int, default=3)
    parser.add_argument("--decoder-bottleneck-channels", type=int, default=48)
    parser.add_argument("--decoder-num-blocks", type=int, default=6)

    # Loss and target preprocessing recovered from the Legacy notebook.
    parser.add_argument("--loss", choices=("gaussian-weighted-mse", "mse"), default="gaussian-weighted-mse")
    parser.add_argument("--loss-peak-start", type=int, default=128)
    parser.add_argument("--loss-peak-end", type=int, default=400)
    parser.add_argument("--loss-peak-weight", type=float, default=3.0)
    parser.add_argument(
        "--zero-diagonal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero the Hi-C main diagonal after log2 transformation, as in the recovered notebooks.",
    )

    # Operational options added for safe long-running retraining.
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--initial-weights", default=None, help="Optional state_dict used only for initialization.")
    initialization.add_argument("--resume", default=None, help="Resume a last_checkpoint.pt produced by this script.")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="Run chromosome-holdout validation every N epochs; the final epoch is always validated.",
    )
    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=None,
        help=(
            "Run only this many training batches, report throughput and peak CUDA memory, then exit "
            "without writing a model checkpoint. Intended for choosing a safe batch size."
        ),
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Limit each split for smoke tests; leave unset for real training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files, sample shapes, resolved settings and model construction without training.",
    )
    return parser.parse_args()


class NotebookCompatibleDataset(Dataset):
    """PKL loader matching the target preprocessing in the recovered notebooks."""

    def __init__(
        self,
        files: Sequence[Path],
        *,
        zero_diagonal: bool,
    ) -> None:
        self.files = list(files)
        self.zero_diagonal = zero_diagonal

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        with self.files[index].open("rb") as handle:
            sample = pickle.load(handle)

        sequence = sample["sequence_one_hot"].astype(np.float32, copy=False)
        omics = sample["omics_signals"].astype(np.float32, copy=False)
        hic = sample["hic_matrix"].astype(np.float32, copy=False)

        hic = np.log2(hic + 1.0).astype(np.float32)
        if self.zero_diagonal:
            hic = hic.copy()
            np.fill_diagonal(hic, 0.0)

        return torch.from_numpy(sequence), torch.from_numpy(omics), torch.from_numpy(hic)


class GaussianWeightedMSELoss(nn.Module):
    """Distance-weighted MSE copied from the closest recovered notebooks."""

    def __init__(self, size: int, peak_range: tuple[int, int], peak_weight: float) -> None:
        super().__init__()
        start, end = peak_range
        if not 0 <= start < end:
            raise ValueError(f"Invalid loss peak range: {peak_range}")
        if peak_weight <= 0:
            raise ValueError("loss peak weight must be positive")

        center = (start + end) / 2.0
        sigma = (end - start) / 4.0
        indices = torch.arange(size, dtype=torch.float32)
        distances = torch.abs(indices[:, None] - indices[None, :])
        gaussian = torch.exp(-((distances - center) ** 2) / (2 * sigma**2))
        self.register_buffer("weights", 1.0 + (peak_weight - 1.0) * gaussian)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((prediction - target) ** 2 * self.weights)


def parse_chromosomes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def filename_has_chromosome(path: Path, chromosome: str) -> bool:
    """Match a chromosome token without confusing chr1 with chr10/chr11."""

    pattern = rf"(^|_){re.escape(chromosome)}(?=_|\.|$)"
    return re.search(pattern, path.name) is not None


def split_files(
    data_dir: Path,
    holdout_chromosome: str,
    excluded_chromosomes: Sequence[str],
    max_samples_per_split: int | None,
) -> tuple[list[Path], list[Path]]:
    all_files = sorted(data_dir.glob("*.pkl"))
    validation_files = [path for path in all_files if filename_has_chromosome(path, holdout_chromosome)]
    training_files = [
        path
        for path in all_files
        if not filename_has_chromosome(path, holdout_chromosome)
        and not any(filename_has_chromosome(path, chromosome) for chromosome in excluded_chromosomes)
    ]
    if max_samples_per_split is not None:
        training_files = training_files[:max_samples_per_split]
        validation_files = validation_files[:max_samples_per_split]
    if not training_files or not validation_files:
        raise ValueError(
            "The chromosome split is empty: "
            f"train={len(training_files)}, validation={len(validation_files)}, "
            f"holdout={holdout_chromosome}, excluded={list(excluded_chromosomes)}"
        )
    return training_files, validation_files


def inspect_sample(path: Path) -> tuple[int, int, dict[str, tuple[int, ...]]]:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    required = ("sequence_one_hot", "omics_signals", "hic_matrix")
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"{path} is missing required fields: {missing}")

    shapes = {key: tuple(sample[key].shape) for key in required}
    sequence_shape = shapes["sequence_one_hot"]
    omics_shape = shapes["omics_signals"]
    hic_shape = shapes["hic_matrix"]
    if len(sequence_shape) != 2 or sequence_shape[0] != 4:
        raise ValueError(f"Expected sequence shape [4, length], got {sequence_shape}")
    if len(omics_shape) != 2:
        raise ValueError(f"Expected omics shape [channels, bins], got {omics_shape}")
    if len(hic_shape) != 2 or hic_shape[0] != hic_shape[1]:
        raise ValueError(f"Expected a square Hi-C matrix, got {hic_shape}")
    if omics_shape[1] != hic_shape[0]:
        raise ValueError(f"Omics/Hi-C bin mismatch: {omics_shape} versus {hic_shape}")
    return omics_shape[0], hic_shape[0], shapes


def create_model(args: argparse.Namespace, num_omics_features: int) -> nn.Module:
    from models import CNNMambaHybridModelLegacy

    if args.fusion_out_dim != args.d_model:
        raise ValueError("The recovered Legacy configuration requires --fusion-out-dim to equal --d-model")
    common = {
        "seq_in_channels": 4,
        "num_omics_features": num_omics_features,
        "d_model": args.d_model,
        "fusion_out_dim": args.fusion_out_dim,
        "num_cnn_blocks": args.num_cnn_blocks,
        "num_mamba_layers": args.num_mamba_layers,
        "decoder_bottleneck_channels": args.decoder_bottleneck_channels,
        "decoder_num_blocks": args.decoder_num_blocks,
    }
    return CNNMambaHybridModelLegacy(**common)


def resolve_profile(args: argparse.Namespace) -> dict:
    return {
        "batch_size": args.batch_size if args.batch_size is not None else LEGACY_PROFILE.batch_size,
        "use_augmentation": False,
        "reverse_complement_probability": 0.0,
        "hic_smoothing_sigma": 0.0,
        "profile": asdict(LEGACY_PROFILE),
    }


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def unwrapped_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def save_loss_weights(criterion: nn.Module, output_dir: Path) -> None:
    if not isinstance(criterion, GaussianWeightedMSELoss):
        return
    weights = criterion.weights.detach().cpu().numpy()
    np.save(output_dir / "loss_weights.npy", weights)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(weights, cmap="viridis")
    axis.set_title("Gaussian distance weights")
    axis.set_xlabel("Hi-C bin")
    axis.set_ylabel("Hi-C bin")
    figure.colorbar(image, ax=axis, label="loss weight")
    figure.tight_layout()
    figure.savefig(output_dir / "loss_weights.png", dpi=160)
    plt.close(figure)


def load_initial_state(model: nn.Module, path: str, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)


def main() -> None:
    args = parse_args()
    resolved = resolve_profile(args)
    if args.max_samples_per_split is not None and args.max_samples_per_split < 1:
        raise ValueError("max samples per split must be positive")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if resolved["batch_size"] < 1:
        raise ValueError("batch size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if args.num_workers < 0:
        raise ValueError("num workers cannot be negative")
    if args.save_every < 0:
        raise ValueError("save every cannot be negative")
    if args.validate_every < 1:
        raise ValueError("validate every must be positive")
    if args.benchmark_batches is not None and args.benchmark_batches < 2:
        raise ValueError("benchmark batches must be at least 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    excluded = (
        parse_chromosomes(args.exclude_chromosomes)
        if args.exclude_chromosomes is not None
        else list(LEGACY_PROFILE.default_excluded_chromosomes)
    )
    training_files, validation_files = split_files(
        data_dir,
        args.holdout_chromosome,
        excluded,
        args.max_samples_per_split,
    )
    num_omics, hic_size, training_shapes = inspect_sample(training_files[0])
    validation_num_omics, validation_hic_size, validation_shapes = inspect_sample(validation_files[0])
    if (validation_num_omics, validation_hic_size, validation_shapes) != (
        num_omics,
        hic_size,
        training_shapes,
    ):
        raise ValueError(
            "Training/validation sample shapes differ: "
            f"train={training_shapes}, validation={validation_shapes}"
        )
    model = create_model(args, num_omics)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    run_summary = {
        "model_type": args.model_type,
        "model_class": LEGACY_PROFILE.model_class,
        "best_model_filename": LEGACY_PROFILE.best_model_filename,
        "profile_provenance": LEGACY_PROFILE.provenance,
        "data_dir": str(data_dir),
        "train_samples": len(training_files),
        "validation_samples": len(validation_files),
        "holdout_chromosome": args.holdout_chromosome,
        "excluded_chromosomes": excluded,
        "sample_shapes": training_shapes,
        "num_omics_features": num_omics,
        "hic_size": hic_size,
        "trainable_parameters": parameter_count,
        "architecture": {
            "d_model": args.d_model,
            "fusion_out_dim": args.fusion_out_dim,
            "num_cnn_blocks": args.num_cnn_blocks,
            "num_mamba_layers": args.num_mamba_layers,
            "decoder_bottleneck_channels": args.decoder_bottleneck_channels,
            "decoder_num_blocks": args.decoder_num_blocks,
        },
        "optimization": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "optimizer": "Adam",
            "loss": args.loss,
            "loss_peak_range": [args.loss_peak_start, args.loss_peak_end],
            "loss_peak_weight": args.loss_peak_weight,
            "zero_diagonal": args.zero_diagonal,
            "num_workers": args.num_workers,
            "validate_every": args.validate_every,
            "seed": args.seed,
        },
        **resolved,
    }
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("Dry run complete; no output directory was created and no training was started.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "training" / "cnn_mamba_legacy"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "run_config.json",
        {"arguments": vars(args), "resolved": run_summary},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError(f"--data-parallel requires at least two visible CUDA GPUs; found {torch.cuda.device_count()}")
        model = nn.DataParallel(model)

    if args.loss == "gaussian-weighted-mse":
        criterion: nn.Module = GaussianWeightedMSELoss(
            hic_size,
            (args.loss_peak_start, args.loss_peak_end),
            args.loss_peak_weight,
        ).to(device)
    else:
        criterion = nn.MSELoss()
    save_loss_weights(criterion, output_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_pcc = float("-inf")
    history: list[dict] = []

    if args.initial_weights:
        load_initial_state(unwrapped_model(model), args.initial_weights, device)
        print(f"Initialized model from {args.initial_weights}")
    elif args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_model_type = checkpoint.get("model_type")
        if checkpoint_model_type is not None and checkpoint_model_type != args.model_type:
            raise ValueError(
                f"Checkpoint model type is {checkpoint_model_type!r}, "
                f"but --model-type is {args.model_type!r}"
            )
        unwrapped_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_pcc = float(checkpoint.get("best_pcc", float("-inf")))
        history = list(checkpoint.get("history", []))
        print(f"Resuming from {args.resume} at epoch {start_epoch + 1}")

    train_dataset = NotebookCompatibleDataset(
        training_files,
        zero_diagonal=args.zero_diagonal,
    )
    validation_dataset = NotebookCompatibleDataset(
        validation_files,
        zero_diagonal=args.zero_diagonal,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=resolved["batch_size"],
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=resolved["batch_size"],
        shuffle=False,
        **loader_options,
    )

    from training_utils import evaluate_model, train_one_epoch

    if args.benchmark_batches is not None:
        model.train()
        warmup_batches = min(10, max(1, args.benchmark_batches // 5))
        measured_samples = 0
        start_time = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch_idx, (seq, omics, hic) in enumerate(train_loader):
            if batch_idx >= args.benchmark_batches:
                break
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
            if batch_idx + 1 == warmup_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                start_time = time.perf_counter()
            elif batch_idx + 1 > warmup_batches:
                measured_samples += int(seq.shape[0])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time if start_time is not None else float("nan")
        result = {
            "batch_size": resolved["batch_size"],
            "num_workers": args.num_workers,
            "total_batches": args.benchmark_batches,
            "warmup_batches": warmup_batches,
            "measured_samples": measured_samples,
            "elapsed_seconds": elapsed,
            "samples_per_second": measured_samples / elapsed,
            "peak_cuda_memory_gib": (
                torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
            ),
        }
        print("BENCHMARK_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        epoch_number = epoch + 1
        should_validate = epoch_number % args.validate_every == 0 or epoch_number == args.epochs
        if should_validate:
            metrics, _, _, _ = evaluate_model(model, validation_loader, device)
            current_pcc = float(metrics.get("mean_pcc", float("nan")))
        else:
            metrics = {}
            current_pcc = float("nan")
        record = {"epoch": epoch_number, "train_loss": train_loss, "validated": should_validate, **metrics}
        history.append(record)
        save_json(output_dir / "history.json", history)

        if np.isfinite(current_pcc) and current_pcc > best_pcc:
            best_pcc = current_pcc
            atomic_torch_save(
                unwrapped_model(model).state_dict(),
                output_dir / LEGACY_PROFILE.best_model_filename,
            )

        checkpoint = {
            "epoch": epoch_number,
            "model_type": args.model_type,
            "model_state_dict": unwrapped_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_pcc": best_pcc,
            "history": history,
            "arguments": vars(args),
            "resolved_profile": resolved,
        }
        atomic_torch_save(checkpoint, output_dir / "last_checkpoint.pt")
        if args.save_every > 0 and epoch_number % args.save_every == 0:
            atomic_torch_save(checkpoint, output_dir / f"checkpoint_epoch_{epoch_number:03d}.pt")
        if should_validate:
            message = (
                f"epoch={epoch_number} train_loss={train_loss:.6f} "
                f"mean_pcc={current_pcc:.6f} best_pcc={best_pcc:.6f}"
            )
        else:
            message = f"epoch={epoch_number} train_loss={train_loss:.6f} validation=skipped best_pcc={best_pcc:.6f}"
        print(message, flush=True)

    print(f"Training complete. Best validation PCC: {best_pcc:.6f}")


if __name__ == "__main__":
    main()
