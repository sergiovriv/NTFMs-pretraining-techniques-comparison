#!/usr/bin/env python3

"""
Quick look and lightweight stats for a *.pt dataset produced by preprocess.py

usage:
    python3 dataset_info.py --dataset_path dataset.pt --peek 3
"""

import argparse, pickle, os, math
from collections import Counter
from pathlib import Path



def describe(item, head=10):
    """Return a short ASCII description of a field inside one sample."""
    if isinstance(item, (list, tuple)):
        preview = " ".join(map(str, item[:head]))
        tail = " ..." if len(item) > head else ""
        return f"list/tuple len={len(item)} {preview}{tail}"
    if hasattr(item, "shape"):      # torch / numpy tensor
        return f"Tensor shape={tuple(item.shape)} dtype={item.dtype}"
    return f"{type(item).__name__}: {item}"

def percentile_from_hist(hist, total, pct):
    """Compute percentile from a Counter histogram without full sort."""
    target = math.ceil(total * pct)
    acc = 0
    for length in sorted(hist):
        acc += hist[length]
        if acc >= target:
            return length
    return None


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset_path", required=True, help="*.pt file from preprocess.py")
    ap.add_argument("--peek", type=int, default=5, help="number of samples to print")
    args = ap.parse_args()

    path = Path(args.dataset_path)
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    print(f"\nFILE  : {path}")
    print(f"SIZE  : {os.path.getsize(path):,} bytes\n")

    samples_to_show = args.peek
    length_hist = Counter()
    total_items = 0

    with path.open("rb") as f:
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break

            total_items += 1

            # assume the first element of the tuple is the input sequence
            if isinstance(item, tuple) and isinstance(item[0], (list, tuple)):
                length_hist[len(item[0])] += 1

            if total_items <= samples_to_show:
                print(f"-- sample {total_items} --")
                if isinstance(item, tuple):
                    for idx, field in enumerate(item):
                        print(f"  [{idx}] {describe(field)}")
                else:
                    print("  value:", describe(item))
                print()

    if length_hist:
        lengths = sorted(length_hist.elements())  
        
        median = percentile_from_hist(length_hist, total_items, 0.50)
        p90    = percentile_from_hist(length_hist, total_items, 0.90)

        print("\nDATASET STATS (based on src length)")
        print(f"instances : {total_items:,}")
        print(f"min len   : {min(length_hist):,}")
        print(f"max len   : {max(length_hist):,}")
        print(f"mean len  : {sum(lengths)/len(lengths):.2f}")
        print(f"median    : {median}")
        print(f"p90       : {p90}")
        mode_len, mode_freq = length_hist.most_common(1)[0]
        print(f"mode len  : {mode_len}  (freq {mode_freq:,})")

        print("\nTOP 20 LENGTHS")
        for length, freq in length_hist.most_common(20):
            print(f"len {length:>5}: {freq:,}")

if __name__ == "__main__":
    main()
