# CNN-Mamba Legacy

This project predicts local bulk Hi-C contact matrices from DNA sequence and four epigenomic tracks. It contains the `CNNMambaHybridModelLegacy` implementation and a supervised training workflow originally configured for GM12878.

The default model takes a 1,024,000 bp sequence and four signals sampled into 512 bins. It predicts a symmetric 512 x 512 contact matrix at 2 kb resolution. The four signal channels must be ordered as **CTCF, DNase, H3K27ac, H3K4me3**.

In this README, **pretraining** means training this supervised model from scratch using measured Hi-C as the target. The repository does not implement a separate self-supervised pretraining objective. Existing compatible weights can also be used for initialization or further training.

The current checkout contains source code; raw datasets, processed samples, and pretrained weights must be supplied or generated. There is no standalone inference script. Section 7 provides an inference example using the existing model API.

## 1. Project Layout and Workflow

```text
cnn_mamba_legacy/
|-- README.md
|-- pyproject.toml
|-- env/
|   |-- setup_training_env.sh
|   `-- verify_training_env.py
|-- preprocessing/
|   |-- pairs_to_2kb_contacts.sh
|   |-- pairs_to_2kb_contacts.awk
|   `-- generate_4omics_pickle.py
|-- src/
|   |-- data.py
|   |-- metrics.py
|   |-- models.py
|   `-- training_utils.py
`-- training/
    `-- train.py
```

The workflow is:

```text
Reference FASTA + four BigWig tracks + filtered Hi-C pairs or contact triplets
    -> 2 kb chromosome contact files
    -> aligned and normalized sliding-window pickle samples
    -> automatic sample validation
    -> supervised training and chromosome-holdout validation
    -> best_model.pth + run_config.json
    -> predictions from DNA and the four normalized signal tracks
```

`src/data.py` provides dataset loaders, `src/models.py` defines the model, `src/metrics.py` implements evaluation metrics, and `src/training_utils.py` contains training and evaluation loops. The main training entry point uses pickle files. An HDF5 loader exists in `src/data.py`, but the supplied generator and training CLI do not provide an HDF5 workflow.

All shell commands below assume **Bash**, an activated environment, and the project root as the current directory. On Windows, use a suitable Linux environment such as WSL for the provided Linux CUDA wheels and shell scripts.

## 2. Environment Requirements

The environment script specifies the following stack:

| Component | Configuration in the script |
|---|---|
| Python | CPython 3.10 |
| Platform for the supplied extension wheels | Linux x86_64 |
| PyTorch | 2.4.1, installed from the CUDA 12.1 wheel index |
| causal-conv1d | 1.4.0 wheel tagged for CUDA 12.2 / PyTorch 2.4 |
| mamba-ssm | 2.2.2 wheel tagged for CUDA 12.2 / PyTorch 2.4 |
| GPU | NVIDIA GPU with a working driver and compatible CUDA extensions |
| Shell utilities for pairs conversion | Bash, `pigz`, `awk`, `find`, `grep`, and `seq` |

The extension wheels must match the Python, platform, PyTorch, and ABI configuration. Run the kernel check below to verify the actual environment. The training code selects a CPU device if CUDA is unavailable, but this does not make the CUDA Mamba implementation a supported CPU workflow.

Python dependencies have different roles:

| Purpose | Packages |
|---|---|
| Model training and evaluation | `torch`, `numpy`, `scipy`, `matplotlib`, `tqdm`, `mamba-ssm`, `causal-conv1d` |
| Python dependencies used by Mamba | `packaging`, `ninja`, `einops`, `triton`, `transformers` |
| Raw-data preprocessing | `numpy`, `scipy`, `pandas`, `pyfaidx`, `pyBigWig` |
| Optional HDF5 loader | `h5py` |

`pyproject.toml` declares the basic dependencies, but does not declare `pandas`, `pyfaidx`, `pyBigWig`, or the Mamba CUDA extensions. Also, `env/setup_training_env.sh` currently references `requirements-training-lock.txt`, which is absent from this checkout. Restore that file before running the setup script, or use the manual installation below.

### Manual installation

The following uses the PyTorch and extension versions selected by the existing setup script. Mamba's additional Python dependencies are listed in its [version 2.2.2 package configuration](https://github.com/state-spaces/mamba/blob/v2.2.2/setup.py). The remaining Python dependencies are not a complete version lock; retain a resolved environment lock for reproducible experiments.

```bash
python3.10 -m venv .venv-training
source .venv-training/bin/activate

python -m pip install --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy scipy matplotlib tqdm h5py pandas pyfaidx pyBigWig
python -m pip install packaging ninja einops triton transformers

python -m pip install --no-deps \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
python -m pip install --no-deps \
  "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

python -m pip install --no-deps --editable .
python -m pip check
python env/verify_training_env.py --kernel-only --gpu 0
```

The kernel check exercises Mamba forward and backward propagation. After preparing samples, run the full check:

```bash
python env/verify_training_env.py \
  --data-dir data/processed/preprocessed_data_omic_512 \
  --gpu 0
```

The full check builds a temporary model and performs one training step with a real sample. It writes `results/environment_validation.json`; it does not create a pretrained checkpoint. This environment check uses ordinary MSE, whereas the main training defaults to the weighted loss described in Section 6.

## 3. Required Raw Data

Use the same reference assembly and chromosome naming convention across FASTA, BigWig, and Hi-C files. The default reference filename is `hg38.fa`, and the default chromosome set is `chr1` through `chr22`. The scripts do not download, align, or lift over data.

Prepare this input layout, or override the corresponding CLI paths:

```text
raw/
|-- dna/
|   `-- hg38.fa
|-- omics/
|   |-- CTCF_ChIP-seq_signal.bigWig
|   |-- DNase_signal.bigWig
|   |-- H3K27ac_ChIP-seq_signal.bigWig
|   `-- H3K4me3_ChIP-seq_signal.bigWig
`-- hic/
    |-- chr1_2kb.txt
    |-- chr2_2kb.txt
    |-- ...
    `-- chr22_2kb.txt
```

### 3.1 Reference genome

Provide a FASTA readable by `pyfaidx`, with chromosome identifiers such as `chr1` and `chr7`. Allow the FASTA index to be created, or provide an existing compatible index. The generator encodes bases in **A, C, G, T** channel order; non-ACGT bases become all-zero columns.

Only complete 2 kb bins are used at the end of each chromosome. Windows are represented by zero-based, half-open genomic intervals: `[start_bp, end_bp)`.

### 3.2 Four epigenomic signals

Provide one BigWig file per track, using the exact filenames above. Each selected chromosome must be available in the reference and signal files. The generator reads the mean signal per 2 kb bin with `pyBigWig`; absent-bin values and non-finite values are replaced with zero before normalization.

For each track, the generator computes a mean and population standard deviation over complete bins on the selected chromosomes, then applies:

```text
normalized_signal = (binned_signal - mean) / std
```

These values are saved in `normalization_stats.json`, ordered as CTCF, DNase, H3K27ac, H3K4me3. The generator rejects near-zero standard deviations when computing statistics.

Keep the normalization file with the trained model. For inference with that model, apply the same track order and training normalization statistics; do not normalize each prediction window independently. Use `--normalization-stats PATH` to reuse statistics in subsequent labeled-data generation runs. `--recompute-stats` explicitly recomputes them.

The default statistics include all selected chromosomes before the training/validation split, including chr7. For an experiment requiring training-only normalization fitting, compute and supply the appropriate statistics separately rather than treating the default as training-only fitting.

### 3.3 Hi-C sparse contact files

Each chromosome file is a tab-separated table with three columns:

| Column | Meaning |
|---|---|
| `pos1_bp` | Zero-based genomic position of the first bin, normally its start |
| `pos2_bp` | Zero-based genomic position of the second bin, normally its start |
| `count` | Nonnegative raw contact count or contact value on the intended target scale |

Example, with literal tab separators:

```text
pos1_bp	pos2_bp	count
0	0	12
0	2000	5
2000	2000	9
2000	4000	3
```

By default, the first line is skipped as a header, positions are divided by 2,000 to obtain bin indices, and the input is treated as a single upper triangle. Duplicate coordinates are summed. Missing entries become zero, and a single triangle is mirrored without doubling its diagonal.

| Input variant | Generator option |
|---|---|
| No header | `--hic-skiprows 0` |
| Bin indices instead of base-pair coordinates | `--hic-coordinate-unit bin` |
| One lower triangle | `--hic-layout lower` |
| Both triangles already present | `--hic-layout full` |
| Detect the stored triangle layout | `--hic-layout auto` |
| Different filenames | `--hic-pattern '{chrom}_contacts.tsv'` |

For `full` layout, the matrix is averaged with its transpose. Do not use the default single-triangle mode on a file containing both triangles, because off-diagonal values would be added twice. The default filename pattern is `{chrom}_2kb.txt`; the generator also tries a `.gz` suffix automatically.

Keep values on the unlogged target scale. Training applies `log2(contact + 1)` itself. No ICE/KR balancing, observed/expected normalization, or library-size normalization is performed by these scripts; any externally chosen target normalization must be consistent across an experiment.

### 3.4 Optional conversion from `.pairs.gz`

If contact triplets are already available, skip this step. Otherwise, provide prefiltered Hi-C pairs with the standard first seven columns:

```text
readID  chrom1  pos1  chrom2  pos2  strand1  strand2
```

The actual file must be tab-separated. Positions are **one-based** in the pairs input. Lines beginning with `#` are skipped. The converter reads chromosome/position columns 2-5; it does not filter mapping quality, pair type, or duplicates, so perform those upstream.

```bash
bash preprocessing/pairs_to_2kb_contacts.sh \
  /path/to/filtered.pairs.gz \
  raw/hic \
  2000 \
  511
```

The converter keeps intra-chromosomal contacts on chr1-chr22, orders the two bins, and retains separations of at most 511 bins (1,022,000 bp between bin starts). This covers all bin-pair distances within a 512-bin window. It writes one upper-triangle contact table per chromosome and checks that all 22 files exist. Use sorted pairs for efficient streaming. The wrapper refuses to overwrite existing `chr*_2kb.txt` files in the output directory.

## 4. Processed Sample Format

The generator writes one Python pickle dictionary per window. A default filename is `chr7_64000_1088000.pkl`, describing the zero-based interval `[64000, 1088000)`.

| Dictionary key | Type / dtype | Default shape | Meaning |
|---|---|---|---|
| `chromosome` | Python `str` | scalar | Chromosome identifier |
| `interval` | Tuple of two integers | `(2,)` | `(start_bp, end_bp)` |
| `sequence_one_hot` | NumPy `uint8` | `(4, 1024000)` | A, C, G, T one-hot sequence |
| `omics_signals` | NumPy `float32` | `(4, 512)` | Normalized CTCF, DNase, H3K27ac, H3K4me3 signals |
| `hic_matrix` | NumPy `float32` | `(512, 512)` | Symmetric, finite, nonnegative contacts before the training log transform |

All three arrays must describe the same interval. Each DNA column must contain at most one `1`, and each matrix row/column must align with the corresponding omics bin. Training converts the input arrays to float32 tensors.

Default sample geometry is:

| Setting | Value |
|---|---|
| Bin size | 2,000 bp |
| Window size | 512 bins = 1,024,000 bp |
| Sliding-window stride | 32 bins = 64,000 bp |
| Maximum non-ACGT fraction | 0.20 per window by default |
| Minimum Hi-C nonzero fraction | 0.0; this additional coverage filter is disabled |

The default `--n-filter-mode window` measures non-ACGT content over the whole window. `--n-filter-mode per-bin` instead rejects a window if any bin exceeds `--max-n-ratio-per-bin`.

Each default sample has approximately 4.91 MiB of array data before pickle overhead. Overlapping windows duplicate data, so plan storage according to the number of accepted windows.

**Model geometry constraint:** the DNA encoder has fixed strides `5 x 5 x 5 x 4 x 4 = 2000`. Preserve 2 kb binning and a DNA length of `2000 x number_of_bins` for the current model. Preprocessing accepts custom sizes, but changing `--bin-size` alone does not reconfigure the encoder. Use the default geometry when reproducing a checkpoint.

## 5. Generate and Validate Samples

Generate the full default dataset:

```bash
python preprocessing/generate_4omics_pickle.py \
  --fasta raw/dna/hg38.fa \
  --omics-dir raw/omics \
  --hic-dir raw/hic \
  --output-dir data/processed/preprocessed_data_omic_512 \
  --bin-size 2000 \
  --window-bins 512 \
  --stride-bins 32
```

The generator bins and normalizes the signals, assembles symmetric Hi-C matrices, filters windows, writes samples, and validates the result. It produces:

```text
data/processed/preprocessed_data_omic_512/
|-- normalization_stats.json
|-- samples.tsv
|-- chr1_START_END.pkl
|-- ...
`-- chr22_START_END.pkl
```

`samples.tsv` contains `chromosome`, `start_bp`, `end_bp`, `file`, `hic_nonzero_ratio`, and `n_ratio`. The last field is the whole-window non-ACGT fraction in window mode, or the largest per-bin fraction in per-bin mode.

Automatic validation checks the complete manifest against the pickle filenames, detects duplicate manifest entries and leftover temporary files, and inspects up to three evenly spaced samples per chromosome. Sample checks cover keys, metadata, interval alignment, shapes, dtypes, one-hot encoding, finite values, and Hi-C symmetry/nonnegativity. Increase `--samples-per-chrom` to inspect more files.

To validate an existing default dataset without loading the raw inputs:

```bash
python preprocessing/generate_4omics_pickle.py \
  --validate-only data/processed/preprocessed_data_omic_512 \
  --samples-per-chrom 10
```

For a small preprocessing check, use a separate output directory:

```bash
python preprocessing/generate_4omics_pickle.py \
  --chromosomes chr7 \
  --max-samples 2 \
  --output-dir data/processed/smoke_chr7

python preprocessing/generate_4omics_pickle.py \
  --validate-only data/processed/smoke_chr7 \
  --chromosomes chr7
```

This chr7-only example checks preprocessing; it cannot supply both training and validation splits. Normalization in this example uses chr7 unless an existing `--normalization-stats` file is supplied. Use the full dataset for training.

`--dry-run` scans inputs and reports accepted windows without writing output files; it still reads raw data and can compute statistics in memory. For validation of custom-size or chromosome-subset datasets, pass the same size and chromosome options used during generation.

Use a fresh output directory when changing generation settings. Existing pickle files are skipped unless `--overwrite` is set; the manifest is rewritten on each generation run. Old files outside the new selection can therefore cause the final manifest check to fail. `--overwrite` replaces matching files but does not remove unrelated old samples.

## 6. Supervised Pretraining and Further Training

### 6.1 Model and target processing

The model has two encoders: one downsamples DNA from base-pair resolution, while the other projects the already binned omics signals. Both apply residual CNN blocks and bidirectional Mamba processing. Their embeddings are concatenated and fused with a 1D convolution. The decoder broadcasts position embeddings into pairwise features, applies 2D residual convolutions, and symmetrizes its output.

Before calculating the loss, the training dataset applies:

```text
target = log2(hic_matrix + 1)
target diagonal = 0, unless --no-zero-diagonal is supplied
```

The default protocol uses no reverse-complement augmentation or Hi-C smoothing. It keeps chr7 for validation and trains on the other available chromosomes, including chr22. Splitting is based on chromosome tokens in filenames. Both splits must contain at least one pickle file.

| Training setting | Default |
|---|---|
| Model | `CNNMambaHybridModelLegacy` |
| Epochs / batch size | 20 / 18 |
| Optimizer / learning rate | Adam / `1e-4` |
| Data loader workers / seed | 8 / 42 |
| Hidden and fusion dimensions | 128 / 128; these must be equal |
| CNN residual blocks / Mamba layers per encoder | 3 / 3 |
| Decoder bottleneck / residual blocks | 48 channels / 6 blocks |
| Loss | Gaussian distance-weighted MSE |
| Validation interval / periodic checkpoint interval | Every epoch / every 5 epochs |

The default loss weight at bin separation `d` is:

```text
w(d) = 1 + 2 * exp(-(d - 264)^2 / (2 * 68^2))
loss = mean(w(abs(i - j)) * (prediction[i, j] - target[i, j])^2)
```

This follows `--loss-peak-start 128`, `--loss-peak-end 400`, and `--loss-peak-weight 3.0`. The weight peaks at separation 264 bins; it is a smooth weighting curve. Use `--loss mse` for ordinary MSE. CUDA training uses automatic mixed precision and gradient scaling.

### 6.2 Check configuration and run a small training job

First inspect the split, sample shapes, resolved settings, and model construction:

```bash
python training/train.py \
  --data-dir data/processed/preprocessed_data_omic_512 \
  --max-samples-per-split 1 \
  --num-workers 0 \
  --dry-run
```

This dry run does not execute a forward pass, load initialization weights, or write training outputs. Use the environment check and a small actual training run to exercise the kernels:

```bash
python training/train.py \
  --data-dir data/processed/preprocessed_data_omic_512 \
  --output-dir results/training/smoke \
  --epochs 1 \
  --batch-size 1 \
  --max-samples-per-split 2 \
  --num-workers 0
```

### 6.3 Train from scratch

```bash
CUDA_VISIBLE_DEVICES=0 python training/train.py \
  --data-dir data/processed/preprocessed_data_omic_512 \
  --output-dir results/training/cnn_mamba_legacy \
  --holdout-chromosome chr7 \
  --epochs 20 \
  --batch-size 18 \
  --learning-rate 1e-4 \
  --num-workers 8
```

Omitting `--initial-weights` and `--resume` starts from random initialization. Batch size 18 is a recovered default, not a GPU-memory guarantee. Reduce it as needed. A throughput and peak-memory check is available:

```bash
python training/train.py \
  --data-dir data/processed/preprocessed_data_omic_512 \
  --output-dir results/training/benchmark \
  --batch-size 1 \
  --benchmark-batches 20
```

Provide enough training batches for warmup and measurement. Benchmark mode updates a temporary model and reports performance without saving a model checkpoint, but still writes run configuration and weighted-loss visualization files.

Use `--exclude-chromosomes chr22` to exclude additional chromosomes from training, or `--holdout-chromosome` to choose another validation chromosome. `--data-parallel` enables `nn.DataParallel` and requires at least two visible CUDA GPUs. Chromosome-holdout validation is used for checkpoint selection; an independent final test set must be organized separately if required.

### 6.4 Initialize from weights or resume a checkpoint

Initialize a new training run with compatible pretrained parameters:

```bash
python training/train.py \
  --data-dir data/processed/preprocessed_data_omic_512 \
  --output-dir results/training/finetuned \
  --initial-weights /path/to/best_model.pth \
  --epochs 10 \
  --learning-rate 1e-5
```

This restores model parameters and creates a fresh optimizer, scaler, and history. It trains all model parameters; no layers are automatically frozen. Match the architecture, input channels, channel order, target processing, and normalization associated with the supplied weights. Loading is strict; the CLI also accepts a checkpoint containing `model_state_dict` for initialization.

Resume an interrupted run with its full checkpoint:

```bash
python training/train.py \
  --data-dir data/processed/preprocessed_data_omic_512 \
  --output-dir results/training/cnn_mamba_legacy \
  --resume results/training/cnn_mamba_legacy/last_checkpoint.pt \
  --epochs 40
```

`--epochs 40` is the total target epoch count, not 40 additional epochs. Resume restores the model, optimizer, mixed-precision scaler, completed epoch, best PCC, and history. Reuse the original data and CLI settings, including architecture and loss settings, because saved arguments are not automatically reapplied. Random-number-generator states are not saved, so resumption is not guaranteed to reproduce an uninterrupted run bit for bit. `--initial-weights` and `--resume` are mutually exclusive.

### 6.5 Training outputs and metrics

The default output directory is `results/training/cnn_mamba_legacy`:

| Output | Contents / purpose |
|---|---|
| `run_config.json` | Command-line arguments, resolved architecture, split counts, sample shapes, and optimization settings |
| `history.json` | Training loss per epoch and metrics on validated epochs |
| `best_model.pth` | Plain model `state_dict`, saved when validation mean Pearson correlation improves |
| `last_checkpoint.pt` | Model, optimizer, scaler, completed epoch, best PCC, history, and configuration metadata |
| `checkpoint_epoch_005.pt`, etc. | Periodic full checkpoints; frequency controlled by `--save-every` |
| `loss_weights.npy` / `loss_weights.png` | Weight matrix and visualization when Gaussian-weighted MSE is used |

Validation computes per-sample Pearson correlation, Spearman correlation, MAE, and MSE, then reports their means and medians. These are calculated on the flattened full matrices in the transformed target space, including both triangles and the diagonal. They are not a distance-stratified SCC. The best checkpoint uses `mean_pcc`, not training loss. `src/metrics.py` also provides distance-dependent PCC and an insulation-score function, but these are not automatically reported by the training CLI.

Keep `best_model.pth`, its matching `run_config.json`, and the dataset's `normalization_stats.json` together for later inference. `best_model.pth` alone does not contain architecture or normalization metadata.

## 7. Inference

### 7.1 Inputs and output scale

Prediction requires **DNA and the four normalized omics tracks**. Ground-truth Hi-C is required for supervised training and optional evaluation, but is not passed to the model during inference.

For a default window, provide float32 tensors of shapes `(B, 4, 1024000)` and `(B, 4, 512)`. The model returns `(B, 512, 512)`. Use the same assembly, genomic alignment, channel order, binning, and training normalization statistics as the checkpoint.

For an unlabeled region, prepare `sequence_one_hot` and `omics_signals` using the encoding, bin means, and saved normalization rules in Sections 3-4. The model itself needs only those two arrays. A feature-only pickle can additionally carry `chromosome` and `interval` metadata for the example below; it need not contain `hic_matrix`. The current generator requires Hi-C files and is intended for labeled sample preparation, so it cannot directly generate unlabeled inference features through its CLI. Do not insert a dummy Hi-C target and treat it as measured data.

The model output is in the learned **log2(contact + 1)** target space. The decoder guarantees symmetry but does not force nonnegative values or an exactly zero diagonal. For a nonnegative contact-like representation, use `maximum(2**prediction - 1, 0)`. This is an inverse transformation of the model output, not a new measured contact matrix or a calibrated count estimate.

### 7.2 Prepare an unlabeled inference window

Skip this step when predicting an already processed training/validation sample. For a new region, the following example reads a reference window and the four BigWig tracks, applies the saved training normalization, and writes a feature-only pickle. It requires no Hi-C input. Change the chromosome, start position, input paths, and normalization path to match the intended experiment.

```bash
python - <<'PY'
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pyBigWig
from pyfaidx import Fasta

root = Path.cwd()
sys.path.insert(0, str(root / "preprocessing"))
from generate_4omics_pickle import TRACK_FILES, encode_one_hot

chromosome = "chr7"
start_bp = 64_000
bin_size = 2_000
window_bins = 512
end_bp = start_bp + bin_size * window_bins
if start_bp < 0 or start_bp % bin_size:
    raise ValueError("The region must start on a nonnegative 2 kb bin boundary")

stats_path = root / "data/processed/preprocessed_data_omic_512/normalization_stats.json"
with stats_path.open(encoding="utf-8") as handle:
    stats = json.load(handle)
if list(stats) != list(TRACK_FILES):
    raise ValueError("Unexpected normalization track order")

fasta = Fasta(str(root / "raw/dna/hg38.fa"), as_raw=True, sequence_always_upper=True)
try:
    if chromosome not in fasta or end_bp > len(fasta[chromosome]):
        raise ValueError("Requested region is outside the reference")
    sequence = str(fasta[chromosome][start_bp:end_bp]).upper()
finally:
    fasta.close()
one_hot = encode_one_hot(sequence)
if np.mean(one_hot.sum(axis=0) == 0) > 0.20:
    raise ValueError("Region exceeds the default non-ACGT fraction threshold")

tracks = []
for track_name, filename in TRACK_FILES.items():
    with pyBigWig.open(str(root / "raw/omics" / filename)) as bw:
        means = bw.stats(chromosome, start_bp, end_bp, nBins=window_bins, type="mean", exact=True)
    values = np.asarray([0.0 if value is None else value for value in means], dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(stats[track_name]["mean"])
    std = float(stats[track_name]["std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-12:
        raise ValueError(f"Invalid saved normalization for {track_name}")
    tracks.append(((values - mean) / std).astype(np.float32))

sample = {
    "chromosome": chromosome,
    "interval": (start_bp, end_bp),
    "sequence_one_hot": one_hot,
    "omics_signals": np.stack(tracks, axis=0),
}
output_dir = root / "data/inference"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{chromosome}_{start_bp}_{end_bp}.pkl"
with output_path.open("wb") as handle:
    pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved inference features: {output_path}")
PY
```

This feature-only format is for inference. The labeled-dataset validator expects `hic_matrix`, so do not pass feature-only files to `--validate-only` or the training CLI.

### 7.3 Predict one window

After a successful default training run, execute the following from the project root. It reads architecture settings from the matching run configuration, loads the best weights, and saves the prediction. By default it selects the first available chr7 sample. To use the unlabeled features from Section 7.2, first run:

```bash
export SAMPLE_PATH="data/inference/chr7_64000_1088000.pkl"
```

Then run the prediction example. Omit `SAMPLE_PATH` to use a labeled chr7 sample instead.

```bash
python - <<'PY'
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
from models import CNNMambaHybridModelLegacy

run_dir = root / "results/training/cnn_mamba_legacy"
data_dir = root / "data/processed/preprocessed_data_omic_512"
if os.environ.get("SAMPLE_PATH"):
    sample_path = Path(os.environ["SAMPLE_PATH"])
else:
    sample_paths = sorted(data_dir.glob("chr7_*.pkl"))
    if not sample_paths:
        raise FileNotFoundError(f"No chr7 sample found in {data_dir}")
    sample_path = sample_paths[0]

with (run_dir / "run_config.json").open(encoding="utf-8") as handle:
    config = json.load(handle)
resolved = config["resolved"]
with sample_path.open("rb") as handle:
    sample = pickle.load(handle)

sequence = np.asarray(sample["sequence_one_hot"], dtype=np.float32)
omics = np.asarray(sample["omics_signals"], dtype=np.float32)
channels = int(resolved["num_omics_features"])
if sequence.ndim != 2 or sequence.shape[0] != 4:
    raise ValueError(f"Invalid DNA shape: {sequence.shape}")
if omics.ndim != 2 or omics.shape[0] != channels:
    raise ValueError(f"Invalid omics shape: {omics.shape}")
if sequence.shape[1] != 2000 * omics.shape[1]:
    raise ValueError("DNA length and 2 kb omics bins do not align")
if not np.isfinite(sequence).all() or not np.isfinite(omics).all():
    raise ValueError("Input arrays must be finite")
if not torch.cuda.is_available():
    raise RuntimeError("A compatible CUDA Mamba environment is required")

device = torch.device("cuda:0")
model = CNNMambaHybridModelLegacy(
    seq_in_channels=4,
    num_omics_features=channels,
    **resolved["architecture"],
)
state = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model = model.to(device).eval()

sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
omics_tensor = torch.from_numpy(omics).unsqueeze(0).to(device)
with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
    prediction = model(sequence_tensor, omics_tensor)
prediction_log2 = prediction.float().cpu().numpy()[0]
prediction_contacts = np.maximum(np.exp2(prediction_log2) - 1.0, 0.0)

output_dir = root / "results/predictions"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{sample_path.stem}.npz"
np.savez_compressed(
    output_path,
    prediction_log2=prediction_log2,
    prediction_contacts=prediction_contacts,
    chromosome=np.asarray(sample["chromosome"]),
    interval=np.asarray(sample["interval"], dtype=np.int64),
    bin_size=np.asarray(2000, dtype=np.int64),
    omics_channel_order=np.asarray(["CTCF", "DNase", "H3K27ac", "H3K4me3"]),
)
print(f"Prediction shape: {prediction_log2.shape}")
print(f"Saved: {output_path}")

# Optional evaluation when the input contains measured Hi-C.
if "hic_matrix" in sample:
    from metrics import batch_sample_metrics

    target = np.log2(np.asarray(sample["hic_matrix"], dtype=np.float32) + 1.0)
    if config["arguments"]["zero_diagonal"]:
        np.fill_diagonal(target, 0.0)
    metrics = batch_sample_metrics(prediction_log2[None], target[None])[0]
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(metrics)
PY
```

The `.npz` file contains the raw log-space prediction, its nonnegative inverse-transformed representation, genomic coordinates, bin size, and channel labels. Matrix index `(i, j)` corresponds to bins starting at `interval[0] + 2000*i` and `interval[0] + 2000*j`.

For multiple windows, load the model once and iterate over windows or batches. The existing `evaluate_model` helper can evaluate labeled datasets, but it retains all predictions and targets in memory. For larger inference jobs, write each batch to disk instead. Overlapping-window stitching, whole-chromosome assembly, `.cool` export, and a batch feature-preparation CLI require additional application code; no such CLI is included here.

## 8. Common Problems

| Symptom | What to check |
|---|---|
| Setup fails at `requirements-training-lock.txt` | The lock file is absent; use the manual installation or restore it. |
| Missing `pyfaidx`, `pyBigWig`, or `pandas` | Install the preprocessing dependencies in the active environment. |
| Mamba import or CUDA extension failure | Check the wheel's Python/platform/PyTorch/ABI match, then run `--kernel-only`. |
| Missing chromosome or mismatched genomic regions | Check reference assembly, `chr` prefixes, and chromosome coverage in all raw inputs. |
| Empty training or validation split | Include both the holdout chromosome and at least one other chromosome; confirm sample filenames. |
| Shape mismatch during fusion | Keep 2 kb bins, matching intervals, and DNA length equal to `2000 x omics bins`. |
| Strict checkpoint load fails | Match architecture options and omics channel count to the checkpoint's run configuration. |
| Manifest/file mismatch after generation | Use a fresh output directory, especially after changing chromosome selections or sample limits. |
| Unexpectedly doubled contact values | Check whether the sparse input contains one triangle or both; set `--hic-layout` accordingly. |
| GPU out of memory | Reduce batch size and measure the configuration with `--benchmark-batches`. |

For the complete command-line options, run `python preprocessing/generate_4omics_pickle.py --help` and `python training/train.py --help` in the configured environment.
