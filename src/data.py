from __future__ import annotations

import glob
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def list_pkl_files(data_dir: str | Path, chromosome: str | None = None, max_samples: int | None = None) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.pkl"))
    if chromosome:
        files = [p for p in files if chromosome in p.name]
    if max_samples is not None:
        files = files[:max_samples]
    return files


def list_h5_files(data_dir: str | Path, chromosome: str | None = None, max_samples: int | None = None) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.h5")) + sorted(Path(data_dir).glob("*.hdf5"))
    if chromosome:
        files = [p for p in files if chromosome in p.name]
    if max_samples is not None:
        files = files[:max_samples]
    return files


def load_sample(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def sample_shapes(path: str | Path) -> dict[str, tuple[int, ...]]:
    data = load_sample(path)
    return {key: tuple(value.shape) for key, value in data.items() if hasattr(value, "shape")}


def parse_indices(indices: str | None) -> list[int] | None:
    if not indices:
        return None
    return [int(part.strip()) for part in indices.split(",") if part.strip()]


def _select_omics(omics: np.ndarray, indices: Sequence[int] | None) -> np.ndarray:
    if indices is None:
        return omics
    return omics[list(indices), :]


def _prepare_hic(hic: np.ndarray, log_hic: bool, zero_diagonal: bool) -> np.ndarray:
    hic = hic.astype(np.float32, copy=False)
    if log_hic:
        hic = np.log2(hic + 1.0).astype(np.float32)
    if zero_diagonal:
        hic = hic.copy()
        np.fill_diagonal(hic, 0)
    return hic


class HiCDataAndOmicsDataset(Dataset):
    """Dataset for preprocessed pickle samples.

    Expected keys are sequence_one_hot, omics_signals, and hic_matrix.
    """

    def __init__(
        self,
        pkl_files: Iterable[str | Path],
        *,
        log_hic: bool = True,
        zero_diagonal: bool = False,
        omics_indices: Sequence[int] | None = None,
    ) -> None:
        self.file_paths = [Path(p) for p in pkl_files]
        self.log_hic = log_hic
        self.zero_diagonal = zero_diagonal
        self.omics_indices = list(omics_indices) if omics_indices is not None else None

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        data = load_sample(self.file_paths[idx])
        sequence = data["sequence_one_hot"].astype(np.float32, copy=False)
        omics = _select_omics(data["omics_signals"].astype(np.float32, copy=False), self.omics_indices)
        hic = _prepare_hic(data["hic_matrix"], self.log_hic, self.zero_diagonal)
        return torch.from_numpy(sequence), torch.from_numpy(omics), torch.from_numpy(hic)


class EfficientHDF5Dataset(Dataset):
    """HDF5 sample dataset with optional omics binning to Hi-C bins."""

    def __init__(
        self,
        hdf5_files: Iterable[str | Path],
        *,
        log_hic: bool = True,
        zero_diagonal: bool = False,
        omics_indices: Sequence[int] | None = None,
        bin_omics: bool = True,
    ) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            raise ImportError("h5py is required for HDF5 datasets.") from exc

        self.hdf5_files = [Path(p) for p in hdf5_files]
        self.log_hic = log_hic
        self.zero_diagonal = zero_diagonal
        self.omics_indices = list(omics_indices) if omics_indices is not None else None
        self.bin_omics = bin_omics
        self.file_handles = [None] * len(self.hdf5_files)
        self.samples: list[tuple[int, int]] = []

        import h5py

        for file_idx, h5_path in enumerate(self.hdf5_files):
            with h5py.File(h5_path, "r") as h5_file:
                for sample_idx in range(len(h5_file["sample_start_bins"][:])):
                    self.samples.append((file_idx, sample_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import h5py

        file_idx, sample_idx = self.samples[idx]
        if self.file_handles[file_idx] is None:
            self.file_handles[file_idx] = h5py.File(self.hdf5_files[file_idx], "r")
        h5_file = self.file_handles[file_idx]

        bin_size = int(h5_file.attrs["bin_size"])
        window_bins = int(h5_file.attrs["window_bins"])
        start_bin = int(h5_file["sample_start_bins"][sample_idx])
        end_bin = start_bin + window_bins
        start_bp = start_bin * bin_size
        end_bp = start_bp + window_bins * bin_size

        sequence = h5_file["sequence_one_hot"][:, start_bp:end_bp].astype(np.float32)
        omics = h5_file["omics_signals"][:, start_bp:end_bp].astype(np.float32)
        omics = _select_omics(omics, self.omics_indices)
        if self.bin_omics:
            channels = omics.shape[0]
            usable = window_bins * bin_size
            omics = omics[:, :usable].reshape(channels, window_bins, bin_size).mean(axis=2).astype(np.float32)
        hic = _prepare_hic(h5_file["hic_matrix"][start_bin:end_bin, start_bin:end_bin], self.log_hic, self.zero_diagonal)
        return torch.from_numpy(sequence), torch.from_numpy(omics), torch.from_numpy(hic)

    def __del__(self) -> None:
        for handle in getattr(self, "file_handles", []):
            if handle is not None:
                handle.close()
