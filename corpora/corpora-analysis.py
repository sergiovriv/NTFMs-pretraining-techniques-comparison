#!/usr/bin/env python3
# corpus_stats_ascii.py
#
# Recorre todos los *.txt de un directorio y muestra:
#   * numero de flows
#   * estadisticos basicos de longitud (tokens por flow)
#   * los TOP‑N largos mas frecuentes
#
# Uso:
#   python3 corpus_stats_ascii.py --dir /ruta/corpora [--top 20]

import argparse, glob, os
from collections import Counter
from math import ceil


def read_histogram(path):
    """Devuelve Counter {longitud: frecuencia}."""
    hist = Counter()
    cur = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():            # fin de flow
                if cur:
                    hist[cur] += 1
                    cur = 0
            else:
                cur += len(line.split())
        if cur:
            hist[cur] += 1
    return hist


def percentiles(hist, total, ps=(25, 50, 75, 90, 95, 99)):
    """Calcula percentiles sin cargar todo en RAM."""
    need = {p: ceil(total * p / 100) for p in ps}
    out = {}
    run = 0
    for length in sorted(hist):
        run += hist[length]
        for p, tgt in need.items():
            if p not in out and run >= tgt:
                out[p] = length
    return out


def stats_for(path, top_n):
    hist = read_histogram(path)
    flows = sum(hist.values())
    tokens = sum(k * v for k, v in hist.items())
    lengths = sorted(hist)

    s = {
        "flows": flows,
        "tokens": tokens,
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(tokens / flows, 2),
        "mode": hist.most_common(1)[0][0],
        "top": hist.most_common(top_n),
    }
    s.update(percentiles(hist, flows))
    return s


def fmt_int(n):
    return f"{n:,}".replace(",", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Carpeta con *.txt")
    ap.add_argument("--top", type=int, default=20, help="N de longitudes en top")
    args = ap.parse_args()

    txts = sorted(glob.glob(os.path.join(args.dir, "*.txt")))
    if not txts:
        raise SystemExit("No se encontraron .txt en " + args.dir)

    print("\nAnalisis de corpus\n")

    for txt in txts:
        s = stats_for(txt, args.top)
        size_mb = os.path.getsize(txt) / 2**20

        print(f"{os.path.basename(txt)}  ({size_mb:,.1f} MB)")
        print(f"  flows:   {fmt_int(s['flows'])}")
        print(f"  len: min {s['min']}  p25 {s[25]}  p50 {s[50]}  p75 {s[75]}  "
              f"p90 {s[90]}  p95 {s[95]}  p99 {s[99]}  max {s['max']}  "
              f"mode {s['mode']}  mean {s['mean']}")
        print("  Top longitudes:")
        for length, freq in s["top"]:
            print(f"    flows de {length:>6} tokens: {fmt_int(freq)}")
        print("-" * 50)


if __name__ == "__main__":
    main()
