#!/usr/bin/env python3
# USAGE: python netbench-to-etbert.py --in_dir <dir> --out <file.txt> [--keep_pkt]

import csv, glob, os, argparse, sys, random, math, subprocess, tempfile, pathlib
from collections import Counter
from datetime import datetime
from tqdm import tqdm

SPECIAL_IGNORE = {"<head>", "</s>"}

def clean(text, keep_pkt):
    out = []
    for tok in text.split():
        t = tok.lower()
        if t in SPECIAL_IGNORE:
            continue
        if t == "<pkt>":
            if keep_pkt:
                out.append("[PKTSEP]")
        else:
            out.append(tok)
    return out

def split_tokens(tok):
    mid = (len(tok) + 1) // 2
    return tok[:mid], tok[mid:]

def percentile(sorted_list, p):
    if not sorted_list:
        return 0.0
    k = (len(sorted_list) - 1) * (p / 100)
    f, c = math.floor(k), math.ceil(k)
    return sorted_list[int(k)] if f == c else sorted_list[f] * (c - k) + sorted_list[c] * (k - f)

def compute_stats(lengths):
    n = len(lengths)
    s = sorted(lengths)
    mean = sum(lengths) / n
    var = sum((x - mean) ** 2 for x in lengths) / n
    std = math.sqrt(var)
    skew = (sum((x - mean) ** 3 for x in lengths) / n) / std ** 3 if std else 0
    kurt = (sum((x - mean) ** 4 for x in lengths) / n) / std ** 4 - 3 if std else 0
    pct = {p: percentile(s, p) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    ctr = Counter(lengths)
    mode, mode_cnt = ctr.most_common(1)[0]
    return {"count": n, "min": s[0], "max": s[-1], "mean": mean, "variance": var,
            "stddev": std, "skewness": skew, "kurtosis": kurt,
            "percentiles": pct, "mode": mode, "mode_count": mode_cnt,
            "top_lengths": ctr.most_common(10)}

def write_stats(stats, corpus_path, size_mb, out_path):
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("CORPUS STATISTICS\n")
        fh.write(f"Generated  : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"Corpus path: {corpus_path}\n")
        fh.write(f"Flows      : {stats['count']:,}\n")
        fh.write(f"Size (MB)  : {size_mb:.1f}\n\n")
        fh.write("Length distribution (tokens per flow):\n")
        fh.write(f"  Min  : {stats['min']}\n")
        fh.write(f"  P50  : {stats['percentiles'][50]:.0f}\n")
        fh.write(f"  P90  : {stats['percentiles'][90]:.0f}\n")
        fh.write(f"  Max  : {stats['max']}\n")
        fh.write(f"  Mean : {stats['mean']:.2f}\n")
        fh.write(f"  Std  : {stats['stddev']:.2f}\n")
        fh.write(f"  Skew : {stats['skewness']:.2f}\n")
        fh.write(f"  Kurt : {stats['kurtosis']:.2f}\n\n")
        fh.write("Top-10 most frequent lengths:\n")
        fh.write(" length | count\n")
        for l, c in stats["top_lengths"]:
            fh.write(f"{l:7d} | {c:7d}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep_pkt", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--temp_dir")
    ap.add_argument("--chunk_pct", type=int, default=10)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("[1/6] Counting train rows...")
    total = 0
    for csv_file in glob.glob(os.path.join(args.in_dir, "*_flow.csv")):
        with open(csv_file, newline='', encoding='utf-8') as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                if row.get("dataset_type", "").strip().lower() == "train":
                    total += 1
    if total == 0:
        sys.exit("No train rows found.")
    print(f"    {total:,} rows selected.")

    tmp_base = args.temp_dir or tempfile.mkdtemp(prefix="netbench_tmp_")
    unsorted_f = os.path.join(tmp_base, "flows.unsorted.txt")
    sorted_f = os.path.join(tmp_base, "flows.sorted.txt")

    print("[2/6] Writing temporary data...")
    with open(unsorted_f, 'w', encoding='utf-8') as fout:
        for fpath in tqdm(sorted(glob.glob(os.path.join(args.in_dir, "*_flow.csv")))):
            with open(fpath, newline='', encoding='utf-8') as fh:
                rdr = csv.DictReader(fh)
                for row in rdr:
                    if row.get("dataset_type", "").strip().lower() != "train":
                        continue
                    toks = clean(row["text"], keep_pkt=args.keep_pkt)
                    if not toks:
                        continue
                    fout.write(f"{random.random():.17f}\t{' '.join(toks)}\n")

    print("[3/6] External sort...")
    subprocess.run(["sort",
                    f"--parallel={args.workers}",
                    "-T", tmp_base,
                    "-n", "-k1,1", unsorted_f,
                    "-o", sorted_f,
                    f"-S{args.chunk_pct}%"],
                   check=True)

    print("[4/6] Generating corpus...")
    lengths = []
    with open(sorted_f, 'r', encoding='utf-8') as fin, \
         open(args.out, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, total=total):
            _, payload = line.split("\t", 1)
            tokens = payload.strip().split()
            lengths.append(len(tokens))
            a, b = split_tokens(tokens)
            fout.write(" ".join(a) + "\n")
            fout.write(" ".join(b) + "\n\n")

    print("[5/6] Computing statistics...")
    stats = compute_stats(lengths)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Corpus : {args.out}")
    print(f"Flows  : {stats['count']:,}")
    print(f"SizeMB : {size_mb:.1f}")
    print(f"Lengths: min {stats['min']} | p50 {stats['percentiles'][50]:.0f} | "
          f"p90 {stats['percentiles'][90]:.0f} | max {stats['max']}")
    print(f"Mean {stats['mean']:.2f} | Std {stats['stddev']:.2f} | Skew {stats['skewness']:.2f}")
    print("Top-5 frequent lengths:")
    for l, c in stats["top_lengths"][:5]:
        print(f"  {l:5d} → {c:,}")
    stats_path = pathlib.Path(args.out).with_suffix(".stats.txt")
    write_stats(stats, args.out, size_mb, stats_path)
    print(f"[6/6] Stats written to {stats_path}")

    if args.temp_dir is None:
        import shutil
        shutil.rmtree(tmp_base)

if __name__ == "__main__":
    main()
