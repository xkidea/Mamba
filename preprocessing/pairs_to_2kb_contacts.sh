#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 INPUT.pairs.gz OUTPUT_DIR [BIN_SIZE=2000] [MAX_SEP_BINS=511]" >&2
    exit 2
fi

input_pairs=$1
output_dir=$2
bin_size=${3:-2000}
max_sep_bins=${4:-511}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -f "$input_pairs" ]]; then
    echo "Input pairs file does not exist: $input_pairs" >&2
    exit 1
fi
if [[ ! "$bin_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "BIN_SIZE must be a positive integer: $bin_size" >&2
    exit 2
fi
if [[ ! "$max_sep_bins" =~ ^[0-9]+$ ]]; then
    echo "MAX_SEP_BINS must be a non-negative integer: $max_sep_bins" >&2
    exit 2
fi

mkdir -p "$output_dir"
if find "$output_dir" -maxdepth 1 -type f -name 'chr*_2kb.txt' -print -quit | grep -q .; then
    echo "Refusing to overwrite existing chr*_2kb.txt files in: $output_dir" >&2
    exit 1
fi

echo "[pairs_to_2kb] input=$input_pairs" >&2
echo "[pairs_to_2kb] output=$output_dir bin_size=$bin_size max_sep_bins=$max_sep_bins" >&2

pigz -dc -- "$input_pairs" | awk \
    -v output_dir="$output_dir" \
    -v bin_size="$bin_size" \
    -v max_sep_bins="$max_sep_bins" \
    -f "$script_dir/pairs_to_2kb_contacts.awk"

for chrom_number in $(seq 1 22); do
    contact_file="$output_dir/chr${chrom_number}_2kb.txt"
    if [[ ! -s "$contact_file" ]]; then
        echo "Missing or empty output: $contact_file" >&2
        exit 1
    fi
done

echo "[pairs_to_2kb] generated 22 autosomal contact files" >&2
