#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CUDA Mamba and one real end-to-end training step."
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "processed" / "preprocessed_data_omic_512"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--kernel-only",
        action="store_true",
        help="Only exercise a standalone Mamba CUDA forward/backward pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    cache_dir = ROOT / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    import causal_conv1d
    import mamba_ssm
    import numpy as np
    import scipy
    import torch
    import torch.nn as nn
    from mamba_ssm import Mamba

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    # Import the compiled modules explicitly so missing/incompatible CUDA wheels
    # fail here instead of halfway through a long training run.
    import causal_conv1d_cuda  # noqa: F401
    import selective_scan_cuda  # noqa: F401

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    # No-argument form targets the current logical CUDA device and is robust
    # when CUDA_VISIBLE_DEVICES remaps a physical GPU to logical cuda:0.
    torch.cuda.reset_peak_memory_stats()

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mamba_ssm": getattr(mamba_ssm, "__version__", "unknown"),
        "causal_conv1d": getattr(causal_conv1d, "__version__", "unknown"),
        "gpu": torch.cuda.get_device_name(device),
    }

    # CUDA-kernel smoke test, including gradients.
    mamba = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).to(device)
    x = torch.randn(1, 512, 128, device=device, requires_grad=True)
    with torch.amp.autocast(device_type="cuda"):
        y = mamba(x)
        kernel_loss = y.square().mean()
    kernel_loss.backward()
    report["mamba_kernel_output"] = list(y.shape)
    report["mamba_kernel_loss"] = float(kernel_loss.detach())
    del x, y, kernel_loss, mamba
    torch.cuda.empty_cache()

    if not args.kernel_only:
        from data import HiCDataAndOmicsDataset, list_pkl_files
        from models import CNNMambaHybridModelLegacy

        files = list_pkl_files(args.data_dir)
        if not files:
            raise FileNotFoundError(f"No .pkl samples found in {args.data_dir}")

        dataset = HiCDataAndOmicsDataset([files[0]])
        sequence, omics, target = dataset[0]
        expected = ((4, 1_024_000), (4, 512), (512, 512))
        actual = (tuple(sequence.shape), tuple(omics.shape), tuple(target.shape))
        if actual != expected:
            raise ValueError(f"Unexpected real-sample shapes: {actual}; expected {expected}")

        model = CNNMambaHybridModelLegacy(num_omics_features=omics.shape[0]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        criterion = nn.MSELoss()

        sequence = sequence.unsqueeze(0).to(device)
        omics = omics.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda"):
            prediction = model(sequence, omics)
            loss = criterion(prediction, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()

        report.update(
            {
                "sample": str(files[0]),
                "sequence_shape": list(sequence.shape),
                "omics_shape": list(omics.shape),
                "target_shape": list(target.shape),
                "prediction_shape": list(prediction.shape),
                "training_loss": float(loss.detach()),
            }
        )

    report["peak_gpu_memory_gib"] = round(
        torch.cuda.max_memory_allocated(device) / (1024**3), 3
    )
    report["status"] = "PASS"

    output = ROOT / "results" / "environment_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Validation report: {output}")


if __name__ == "__main__":
    main()
