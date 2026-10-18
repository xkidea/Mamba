BEGIN {
    FS = OFS = "\t"
    if (bin_size == "") bin_size = 2000
    if (max_sep_bins == "") max_sep_bins = 511
    if (output_dir == "") output_dir = "."
    current_chrom = ""
    current_bin1 = -1
    data_rows = 0
    kept_rows = 0
}

function is_autosome(chrom) {
    return chrom ~ /^chr([1-9]|1[0-9]|2[0-2])$/
}

function flush_bin1(    delta, bin2, path) {
    if (current_chrom == "" || current_bin1 < 0) return
    path = output_dir "/" current_chrom "_2kb.txt"
    for (delta = 0; delta <= max_sep_bins; delta++) {
        bin2 = current_bin1 + delta
        if (counts[bin2] > 0) {
            print current_bin1 * bin_size, bin2 * bin_size, counts[bin2] >> path
            delete counts[bin2]
        }
    }
}

/^#/ { next }

{
    data_rows++
    if (data_rows % 50000000 == 0) {
        printf("[pairs_to_2kb] read=%d kept=%d chrom=%s bin1=%d\n",
               data_rows, kept_rows, current_chrom, current_bin1) > "/dev/stderr"
    }

    chrom1 = $2
    chrom2 = $4
    if (chrom1 != chrom2 || !is_autosome(chrom1)) next

    # pairs format v1.0 uses 1-based genomic positions.
    bin1 = int(($3 - 1) / bin_size)
    bin2 = int(($5 - 1) / bin_size)
    if (bin2 < bin1) {
        tmp = bin1
        bin1 = bin2
        bin2 = tmp
    }
    if (bin2 - bin1 > max_sep_bins) next

    if (chrom1 != current_chrom || bin1 != current_bin1) {
        flush_bin1()
        current_chrom = chrom1
        current_bin1 = bin1
        path = output_dir "/" current_chrom "_2kb.txt"
        if (!initialized[current_chrom]) {
            print "pos1_bp", "pos2_bp", "count" > path
            initialized[current_chrom] = 1
        }
    }

    counts[bin2]++
    kept_rows++
}

END {
    flush_bin1()
    printf("[pairs_to_2kb] completed read=%d kept=%d\n", data_rows, kept_rows) > "/dev/stderr"
}
