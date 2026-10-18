#!/usr/bin/env python3
"""Generate and validate the four-omics pickle dataset used by MambaHiC.

Each output pickle is a plain dict with the same keys and dtypes observed in the
existing standard dataset:

    chromosome: str
    interval: (start_bp, end_bp)
    sequence_one_hot: uint8 [4, 1_024_000]
    hic_matrix: float32 [512, 512] (raw contacts; training applies log2(x + 1))
    omics_signals: float32 [4, 512]

The omics channel order is fixed to CTCF, DNase, H3K27ac, H3K4me3.
Generated datasets are validated automatically. To check an existing dataset,
use --validate-only DATA_DIR, optionally with --chromosomes and size options.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"^(chr[^_]+)_(\d+)_(\d+)\.pkl$")
EXPECTED_KEYS = {
    "chromosome",
    "interval",
    "sequence_one_hot",
    "hic_matrix",
    "omics_signals",
}
TRACK_FILES = OrderedDict(
    [
        ("CTCF", "CTCF_ChIP-seq_signal.bigWig"),
        ("DNase", "DNase_signal.bigWig"),
        ("H3K27ac", "H3K27ac_ChIP-seq_signal.bigWig"),
        ("H3K4me3", "H3K4me3_ChIP-seq_signal.bigWig"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and validate four-channel 2-kb/512-bin pickle samples."
    )
    parser.add_argument("--fasta", type=Path, default=ROOT / "raw/dna/hg38.fa")
    parser.add_argument("--omics-dir", type=Path, default=ROOT / "raw/omics")
    parser.add_argument(
        "--hic-dir",
        type=Path,
        default=ROOT / "raw/hic",
        help="Directory containing chrN_2kb.txt or chrN_2kb.txt.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/preprocessed_data_omic_512",
    )
    parser.add_argument(
        "--chromosomes",
        default=",".join(f"chr{i}" for i in range(1, 23)),
        help="Comma-separated chromosomes. The recovered standard dataset contains chr1-chr22, not chrX.",
    )
    parser.add_argument("--bin-size", type=int, default=2000)
    parser.add_argument("--window-bins", type=int, default=512)
    parser.add_argument(
        "--stride-bins",
        type=int,
        default=32,
        help="32 bins = 64 kb, matching the recovered 42,005-file dataset.",
    )
    parser.add_argument(
        "--max-n-ratio-per-bin",
        type=float,
        default=0.20,
        help="Maximum non-ACGT fraction used by the selected N-filter mode.",
    )
    parser.add_argument(
        "--n-filter-mode",
        choices=("window", "per-bin"),
        default="window",
        help="The recovered standard PKLs fit a whole-window filter better; per-bin matches the later HDF5 draft's intent.",
    )
    parser.add_argument(
        "--min-hic-nonzero-ratio",
        type=float,
        default=0.0,
        help="Optional additional low-coverage filter. Zero disables it because the original threshold was not recovered.",
    )
    parser.add_argument(
        "--hic-pattern",
        default="{chrom}_2kb.txt",
        help="Filename pattern. A .gz suffix is also tried automatically.",
    )
    parser.add_argument(
        "--hic-skiprows",
        type=int,
        default=1,
        help="The recovered notebook skipped the first line as a header.",
    )
    parser.add_argument(
        "--hic-layout",
        choices=("upper", "lower", "full", "auto"),
        default="upper",
        help="Sparse matrix layout. The recovered notebook treated input as one triangle.",
    )
    parser.add_argument(
        "--hic-coordinate-unit",
        choices=("bp", "bin"),
        default="bp",
        help="The recovered files used base-pair coordinates in columns 1 and 2.",
    )
    parser.add_argument("--read-chunksize", type=int, default=2_000_000)
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=None,
        help="JSON path for reusable genome-wide omics mean/std values.",
    )
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Useful for a smoke test.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Scan and report samples without writing pickle files.")
    mode.add_argument(
        "--validate-only",
        type=Path,
        metavar="DATA_DIR",
        help="Validate existing samples without reading raw inputs or generating files.",
    )
    parser.add_argument(
        "--samples-per-chrom",
        type=int,
        default=3,
        help="Number of pickle files checked per chromosome after generation or with --validate-only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def comma_list(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("No chromosomes were selected.")
    return result


def require_inputs(args: argparse.Namespace, chromosomes: Iterable[str]) -> dict[str, Path]:
    if not args.fasta.is_file():
        raise FileNotFoundError(f"FASTA not found: {args.fasta}")
    track_paths = {name: args.omics_dir / filename for name, filename in TRACK_FILES.items()}
    missing = [str(path) for path in track_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing BigWig files:\n" + "\n".join(missing))
    for chrom in chromosomes:
        resolve_hic_path(args.hic_dir, args.hic_pattern, chrom)
    return track_paths


def resolve_hic_path(hic_dir: Path, pattern: str, chrom: str) -> Path:
    path = hic_dir / pattern.format(chrom=chrom)
    candidates = [path]
    if not str(path).endswith(".gz"):
        candidates.append(Path(str(path) + ".gz"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Hi-C file not found for {chrom}; tried: {candidates}")


def binned_bigwig_values(bw, chrom: str, chrom_len: int, bin_size: int) -> np.ndarray:
    n_bins = chrom_len // bin_size
    if n_bins <= 0:
        return np.empty(0, dtype=np.float64)
    end_bp = n_bins * bin_size
    values = bw.stats(chrom, 0, end_bp, nBins=n_bins, type="mean", exact=True)
    return np.asarray([0.0 if value is None else value for value in values], dtype=np.float64)


def compute_global_stats(
    fasta,
    track_paths: dict[str, Path],
    chromosomes: list[str],
    bin_size: int,
) -> dict[str, dict[str, float]]:
    import pyBigWig

    stats: dict[str, dict[str, float]] = {}
    for track_name, path in track_paths.items():
        total = 0.0
        total_sq = 0.0
        count = 0
        with pyBigWig.open(str(path)) as bw:
            bw_chroms = bw.chroms()
            for chrom in chromosomes:
                if chrom not in fasta or chrom not in bw_chroms:
                    raise KeyError(f"{chrom} is missing from FASTA or {path}")
                values = binned_bigwig_values(bw, chrom, len(fasta[chrom]), bin_size)
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                total += float(values.sum(dtype=np.float64))
                total_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
                count += int(values.size)
        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        std = math.sqrt(variance)
        if std < 1e-12:
            raise ValueError(f"Near-zero standard deviation for {track_name}: {std}")
        stats[track_name] = {"mean": mean, "std": std, "n_bins": count}
        print(f"stats {track_name}: mean={mean:.8g}, std={std:.8g}, bins={count}")
    return stats


def load_or_compute_stats(
    args: argparse.Namespace,
    fasta,
    track_paths: dict[str, Path],
    chromosomes: list[str],
) -> tuple[dict[str, dict[str, float]], Path]:
    stats_path = args.normalization_stats or (args.output_dir / "normalization_stats.json")
    if stats_path.is_file() and not args.recompute_stats:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if list(stats) != list(TRACK_FILES):
            raise ValueError(f"Unexpected track order in {stats_path}: {list(stats)}")
        return stats, stats_path
    stats = compute_global_stats(fasta, track_paths, chromosomes, args.bin_size)
    if not args.dry_run:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats, stats_path


def load_hic_matrix(
    path: Path,
    n_bins: int,
    *,
    bin_size: int,
    skiprows: int,
    chunksize: int,
    coordinate_unit: str,
    layout: str,
):
    import pandas as pd
    from scipy import sparse

    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    values_all: list[np.ndarray] = []
    upper_count = 0
    lower_count = 0
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        skiprows=skiprows,
        usecols=[0, 1, 2],
        chunksize=chunksize,
        compression="infer",
    )
    for chunk in reader:
        rows = pd.to_numeric(chunk.iloc[:, 0], errors="coerce").to_numpy()
        cols = pd.to_numeric(chunk.iloc[:, 1], errors="coerce").to_numpy()
        values = pd.to_numeric(chunk.iloc[:, 2], errors="coerce").to_numpy(dtype=np.float64)
        good = np.isfinite(rows) & np.isfinite(cols)
        rows = rows[good].astype(np.int64, copy=False)
        cols = cols[good].astype(np.int64, copy=False)
        values = np.nan_to_num(values[good], nan=0.0, posinf=0.0, neginf=0.0)
        if coordinate_unit == "bp":
            rows //= bin_size
            cols //= bin_size
        in_range = (rows >= 0) & (rows < n_bins) & (cols >= 0) & (cols < n_bins)
        rows = rows[in_range]
        cols = cols[in_range]
        values = values[in_range]
        upper_count += int(np.count_nonzero(rows < cols))
        lower_count += int(np.count_nonzero(rows > cols))
        rows_all.append(rows)
        cols_all.append(cols)
        values_all.append(values.astype(np.float32, copy=False))
    if not rows_all:
        raise ValueError(f"No Hi-C records parsed from {path}")
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    values = np.concatenate(values_all)
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(n_bins, n_bins), dtype=np.float32).tocsr()
    matrix.sum_duplicates()

    effective_layout = layout
    if layout == "auto":
        effective_layout = "upper" if lower_count == 0 else "lower" if upper_count == 0 else "full"
    if effective_layout in {"upper", "lower"}:
        # Mirror a single stored triangle without doubling diagonal entries.
        matrix = matrix + matrix.T - sparse.diags(matrix.diagonal(), format="csr")
    elif effective_layout == "full":
        # Ensure exact symmetry while retaining scale if both triangles are present.
        matrix = ((matrix + matrix.T) * 0.5).tocsr()
    matrix.eliminate_zeros()
    print(
        f"loaded {path.name}: shape={matrix.shape}, nnz={matrix.nnz}, "
        f"upper_records={upper_count}, lower_records={lower_count}, layout={effective_layout}"
    )
    return matrix


def encode_one_hot(sequence: str) -> np.ndarray:
    encoded = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
    output = np.zeros((4, encoded.size), dtype=np.uint8)
    for channel, base in enumerate(b"ACGT"):
        output[channel, encoded == base] = 1
    return output


def sequence_bin_invalid_counts(sequence: str, bin_size: int) -> np.ndarray:
    encoded = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
    usable = (encoded.size // bin_size) * bin_size
    encoded = encoded[:usable].reshape(-1, bin_size)
    valid = (encoded == ord("A")) | (encoded == ord("C")) | (encoded == ord("G")) | (encoded == ord("T"))
    return (~valid).sum(axis=1, dtype=np.int32)


def chromosome_omics(
    track_paths: dict[str, Path],
    stats: dict[str, dict[str, float]],
    chrom: str,
    chrom_len: int,
    bin_size: int,
) -> np.ndarray:
    import pyBigWig

    tracks = []
    for track_name, path in track_paths.items():
        with pyBigWig.open(str(path)) as bw:
            values = binned_bigwig_values(bw, chrom, chrom_len, bin_size)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean = float(stats[track_name]["mean"])
        std = float(stats[track_name]["std"])
        tracks.append(((values - mean) / std).astype(np.float32))
    return np.stack(tracks, axis=0)


def validate_sample(sample: dict, window_bins: int, bin_size: int) -> None:
    expected_bp = window_bins * bin_size
    expected = {
        "sequence_one_hot": ((4, expected_bp), np.dtype("uint8")),
        "hic_matrix": ((window_bins, window_bins), np.dtype("float32")),
        "omics_signals": ((4, window_bins), np.dtype("float32")),
    }
    for key, (shape, dtype) in expected.items():
        value = sample[key]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{key}: got shape={value.shape}, dtype={value.dtype}; expected {shape}, {dtype}")
    if not np.array_equal(sample["hic_matrix"], sample["hic_matrix"].T):
        raise ValueError("Hi-C window is not symmetric.")


def validate_pickle(
    path: Path,
    *,
    window_bins: int = 512,
    bin_size: int = 2000,
    stride_bins: int = 32,
) -> None:
    match = NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected pickle filename: {path.name}")
    expected_chrom = match.group(1)
    expected_interval = (int(match.group(2)), int(match.group(3)))
    if expected_interval[1] - expected_interval[0] != window_bins * bin_size:
        raise ValueError(f"Unexpected interval length: {path.name}")
    if expected_interval[0] % (stride_bins * bin_size) != 0:
        raise ValueError(f"Start is not aligned to the {stride_bins}-bin stride: {path.name}")

    with path.open("rb") as handle:
        sample = pickle.load(handle)
    if set(sample) != EXPECTED_KEYS:
        raise ValueError(f"Unexpected keys in {path.name}: {list(sample)}")
    if sample["chromosome"] != expected_chrom or tuple(sample["interval"]) != expected_interval:
        raise ValueError(f"Metadata/filename mismatch: {path.name}")
    validate_sample(sample, window_bins, bin_size)

    sequence = sample["sequence_one_hot"]
    hic = sample["hic_matrix"]
    omics = sample["omics_signals"]
    if not np.all((sequence == 0) | (sequence == 1)) or np.any(sequence.sum(axis=0) > 1):
        raise ValueError(f"Invalid one-hot encoding in {path.name}")
    if not np.isfinite(hic).all() or np.any(hic < 0):
        raise ValueError(f"Invalid Hi-C values in {path.name}")
    if not np.isfinite(np.log2(hic + 1.0)).all():
        raise ValueError(f"Non-finite training Hi-C transform in {path.name}")
    if not np.isfinite(omics).all():
        raise ValueError(f"Non-finite omics values in {path.name}")


def validate_generated_pickles(
    data_dir: Path,
    *,
    chromosomes: Iterable[str],
    samples_per_chrom: int = 3,
    window_bins: int = 512,
    bin_size: int = 2000,
    stride_bins: int = 32,
) -> None:
    """Check the full manifest and inspect evenly spaced samples per chromosome."""
    if min(samples_per_chrom, window_bins, bin_size, stride_bins) < 1:
        raise ValueError("Validation sample count and size options must be positive.")
    paths = sorted(data_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No pickle files found: {data_dir}")
    temporary = list(data_dir.glob("*.tmp"))
    if temporary:
        raise ValueError(f"Temporary files remain: {temporary[:5]}")

    with (data_dir / "samples.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    manifest_names = [row["file"] for row in rows]
    actual_names = [path.name for path in paths]
    if len(manifest_names) != len(set(manifest_names)):
        raise ValueError("Duplicate filenames in samples.tsv")
    if set(manifest_names) != set(actual_names):
        missing = set(manifest_names) - set(actual_names)
        extra = set(actual_names) - set(manifest_names)
        raise ValueError(f"Manifest/file mismatch: missing={len(missing)}, extra={len(extra)}")

    by_chrom: dict[str, list[Path]] = {}
    for path in paths:
        match = NAME_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected pickle filename: {path.name}")
        by_chrom.setdefault(match.group(1), []).append(path)
    expected_chroms = list(dict.fromkeys(chromosomes))
    if set(by_chrom) != set(expected_chroms):
        raise ValueError(f"Chromosome set mismatch: expected={expected_chroms}, actual={sorted(by_chrom)}")

    selected: list[Path] = []
    for chrom in expected_chroms:
        chrom_paths = by_chrom[chrom]
        indices = np.linspace(0, len(chrom_paths) - 1, samples_per_chrom, dtype=int)
        selected.extend(chrom_paths[index] for index in sorted(set(indices.tolist())))
    for path in selected:
        validate_pickle(path, window_bins=window_bins, bin_size=bin_size, stride_bins=stride_bins)

    counts = Counter(row["chromosome"] for row in rows)
    print(f"validated files={len(paths)} manifest_rows={len(rows)} sampled_pickles={len(selected)}")
    print("chromosome_counts=" + ",".join(f"{chrom}:{counts[chrom]}" for chrom in expected_chroms))
    sizes = np.fromiter((path.stat().st_size for path in paths), dtype=np.int64)
    print(f"file_size_bytes min={sizes.min()} max={sizes.max()} total={sizes.sum()}")


def write_pickle(path: Path, sample: dict, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(path)
    return True


def process_chromosome(
    args: argparse.Namespace,
    fasta,
    track_paths: dict[str, Path],
    stats: dict[str, dict[str, float]],
    chrom: str,
    manifest_writer,
    remaining: int | None,
) -> tuple[int, int]:
    chrom_len = len(fasta[chrom])
    n_bins = chrom_len // args.bin_size
    usable_bp = n_bins * args.bin_size
    full_sequence = str(fasta[chrom][:usable_bp]).upper()
    invalid_per_bin = sequence_bin_invalid_counts(full_sequence, args.bin_size)
    omics = chromosome_omics(track_paths, stats, chrom, chrom_len, args.bin_size)
    if omics.shape != (4, n_bins):
        raise ValueError(f"Unexpected omics shape for {chrom}: {omics.shape}; expected {(4, n_bins)}")

    hic_path = resolve_hic_path(args.hic_dir, args.hic_pattern, chrom)
    hic = load_hic_matrix(
        hic_path,
        n_bins,
        bin_size=args.bin_size,
        skiprows=args.hic_skiprows,
        chunksize=args.read_chunksize,
        coordinate_unit=args.hic_coordinate_unit,
        layout=args.hic_layout,
    )

    accepted = 0
    written = 0
    for start_bin in range(0, n_bins - args.window_bins + 1, args.stride_bins):
        end_bin = start_bin + args.window_bins
        window_invalid = invalid_per_bin[start_bin:end_bin]
        if args.n_filter_mode == "per-bin":
            reject_for_n = np.any(window_invalid / args.bin_size > args.max_n_ratio_per_bin)
            reported_n_ratio = float(window_invalid.max(initial=0) / args.bin_size)
        else:
            reported_n_ratio = float(window_invalid.sum() / (args.window_bins * args.bin_size))
            reject_for_n = reported_n_ratio > args.max_n_ratio_per_bin
        if reject_for_n:
            continue
        hic_window = hic[start_bin:end_bin, start_bin:end_bin].toarray().astype(np.float32, copy=False)
        hic_nonzero_ratio = float(np.count_nonzero(hic_window) / hic_window.size)
        if hic_nonzero_ratio < args.min_hic_nonzero_ratio:
            continue
        start_bp = start_bin * args.bin_size
        end_bp = end_bin * args.bin_size
        output_path = args.output_dir / f"{chrom}_{start_bp}_{end_bp}.pkl"
        accepted += 1
        if not args.dry_run:
            sequence = full_sequence[start_bp:end_bp]
            sample = {
                "chromosome": chrom,
                "interval": (start_bp, end_bp),
                "sequence_one_hot": encode_one_hot(sequence),
                "hic_matrix": hic_window,
                "omics_signals": omics[:, start_bin:end_bin].astype(np.float32, copy=False),
            }
            validate_sample(sample, args.window_bins, args.bin_size)
            did_write = write_pickle(output_path, sample, args.overwrite)
            written += int(did_write)
            manifest_writer.writerow(
                [chrom, start_bp, end_bp, output_path.name, f"{hic_nonzero_ratio:.8g}", f"{reported_n_ratio:.8g}"]
            )
        if remaining is not None and accepted >= remaining:
            break
    print(f"{chrom}: accepted={accepted}, newly_written={written}")
    return accepted, written


def main() -> int:
    args = parse_args()
    chromosomes = comma_list(args.chromosomes)
    if args.bin_size <= 0 or args.window_bins <= 0 or args.stride_bins <= 0:
        raise ValueError("bin-size, window-bins, and stride-bins must be positive.")
    if args.samples_per_chrom < 1:
        raise ValueError("samples-per-chrom must be positive.")
    if not 0.0 <= args.max_n_ratio_per_bin <= 1.0:
        raise ValueError("max-n-ratio-per-bin must be between 0 and 1.")
    if not 0.0 <= args.min_hic_nonzero_ratio <= 1.0:
        raise ValueError("min-hic-nonzero-ratio must be between 0 and 1.")

    if args.validate_only is not None:
        validate_generated_pickles(
            args.validate_only,
            chromosomes=chromosomes,
            samples_per_chrom=args.samples_per_chrom,
            window_bins=args.window_bins,
            bin_size=args.bin_size,
            stride_bins=args.stride_bins,
        )
        return 0

    from pyfaidx import Fasta

    track_paths = require_inputs(args, chromosomes)
    fasta = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)
    missing_chroms = [chrom for chrom in chromosomes if chrom not in fasta]
    if missing_chroms:
        raise KeyError(f"Chromosomes missing from FASTA: {missing_chroms}")
    stats, stats_path = load_or_compute_stats(args, fasta, track_paths, chromosomes)
    print(f"normalization stats: {stats_path}")
    print("omics order:", ", ".join(TRACK_FILES))

    manifest_handle = None
    if args.dry_run:
        manifest_writer = csv.writer(sys.stdout)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_handle = (args.output_dir / "samples.tsv").open("w", newline="", encoding="utf-8")
        manifest_writer = csv.writer(manifest_handle, delimiter="\t")
        manifest_writer.writerow(["chromosome", "start_bp", "end_bp", "file", "hic_nonzero_ratio", "n_ratio"])

    total_accepted = 0
    total_written = 0
    generated_chromosomes = []
    try:
        for chrom in chromosomes:
            remaining = None if args.max_samples is None else args.max_samples - total_accepted
            if remaining is not None and remaining <= 0:
                break
            accepted, written = process_chromosome(
                args, fasta, track_paths, stats, chrom, manifest_writer, remaining
            )
            total_accepted += accepted
            total_written += written
            if accepted:
                generated_chromosomes.append(chrom)
    finally:
        if manifest_handle is not None:
            manifest_handle.close()
        fasta.close()

    print(f"complete: accepted={total_accepted}, newly_written={total_written}, output={args.output_dir}")
    if total_accepted == 0:
        raise RuntimeError("No samples passed the filters.")
    if not args.dry_run:
        validate_generated_pickles(
            args.output_dir,
            chromosomes=generated_chromosomes,
            samples_per_chrom=args.samples_per_chrom,
            window_bins=args.window_bins,
            bin_size=args.bin_size,
            stride_bins=args.stride_bins,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
