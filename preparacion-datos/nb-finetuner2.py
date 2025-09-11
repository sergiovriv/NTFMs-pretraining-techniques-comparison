#!/usr/bin/env python3

from __future__ import annotations
import os, random
from datetime import datetime
import pandas as pd
from collections import defaultdict
from typing import List, Dict


def csv_files(path: str) -> List[str]:
    return [f for f in os.listdir(path) if f.lower().endswith(".csv")]


def choose(opts: List[str], msg: str) -> str:
    for i, o in enumerate(opts):
        print(f"[{i}] {o}")
    while True:
        x = input(msg).strip()
        if x.isdigit() and int(x) in range(len(opts)):
            return opts[int(x)]
        print("Selección inválida.")


def ask_max() -> int | None:
    x = input("Máximo flows (solo TRAIN, ENTER = sin límite): ").strip()
    return int(x) if x.isdigit() and int(x) > 0 else None


def balance(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    groups: Dict[int, List[int]] = {}
    for lab, idx in df.groupby("label"):
        lst = list(idx.index)
        random.shuffle(lst)           # barajado simple
        groups[lab] = lst

    total = sum(len(v) for v in groups.values())
    lim   = total if limit is None else min(limit, total)

    pick: List[int] = []
    cycle = list(groups.keys())       # orden fijo de clases
    while len(pick) < lim and cycle:
        for lab in list(cycle):
            if groups[lab]:
                pick.append(groups[lab].pop(0))
                if len(pick) == lim:
                    break
            else:
                cycle.remove(lab)
    return df.loc[pick].reset_index(drop=True)


def main() -> None:
    csvs = csv_files(os.getcwd())
    if not csvs:
        print("Sin CSV aquí.")
        return

    csv_file = choose(csvs, "Elige CSV: ")
    df = pd.read_csv(csv_file)

    if {"text", "dataset_type"} - set(df.columns):
        print("Faltan columnas 'text' y/o 'dataset_type'.")
        return

    keep_pkt = input("Conservar [PKTSEP] a partir de <pkt>? (s/N): ").strip().lower().startswith("s")

    if keep_pkt:
        df["text"] = (df["text"].astype(str)
                      .str.replace(r"(?i)<pkt>", "[PKTSEP]", regex=True)  # solo <pkt> -> [PKTSEP]
                      .str.replace(r"<[^>]+>", "", regex=True)            
                      .str.replace(r"\s+", " ", regex=True)
                      .str.strip())
    else:
        df["text"] = (df["text"].astype(str)
                      .str.replace(r"<[^>]+>", "", regex=True)           
                      .str.replace(r"\s+", " ", regex=True)
                      .str.strip())

    label_cands = df.columns[df.columns.get_loc("dataset_type") + 1:]
    if not len(label_cands):
        print("No hay columnas de etiqueta.")
        return
    label_col = choose(label_cands.tolist(), "Elige columna etiqueta: ")

    max_flows = ask_max()

    out_dir = input("Carpeta salida: ").strip()
    os.makedirs(out_dir, exist_ok=True)

    df = df[["text", "dataset_type", label_col]].rename(
        columns={"text": "text_a", label_col: "label"})
    df = df[df["label"].notna()].reset_index(drop=True)

    labels = sorted(df["label"].unique())
    lab2id = {lab: i for i, lab in enumerate(labels)}
    id2lab = {i: lab for lab, i in lab2id.items()}

    split_files = {"train": "train_dataset.tsv",
                   "dev":   "valid_dataset.tsv",
                   "test":  "test_dataset.tsv"}

    stats_split, stats_lab = [], defaultdict(int)

    # Acumuladores para el informe fusionado
    per_split_counts: Dict[str, pd.Series] = {}
    per_split_lengths: Dict[str, pd.Series] = {}
    per_split_missing: Dict[str, List[int]] = {}

    for split, fname in split_files.items():
        part = df[df["dataset_type"] == split].copy()
        if part.empty:
            continue
        part["label"] = part["label"].map(lab2id)

        limit = max_flows if split == "train" else None
        part  = balance(part, limit)

        n = len(part)
        stats_split.append((split, n))
        counts = part["label"].value_counts().sort_index()
        per_split_counts[split] = counts
        for k, v in counts.items():
            stats_lab[k] += int(v)

        present = set(counts.index.tolist())
        all_ids = set(range(len(labels)))
        per_split_missing[split] = sorted(all_ids - present)

        lens = part["text_a"].astype(str).str.split().map(len)
        per_split_lengths[split] = lens

        part[["label", "text_a"]].to_csv(
            os.path.join(out_dir, fname), sep="\t", index=False)

        if split == "test":
            part[["text_a"]].to_csv(
                os.path.join(out_dir, "nolabel_test_dataset.tsv"),
                sep="\t", index=False)

    total = sum(stats_lab.values())
    with open(os.path.join(out_dir, "label_mapping.txt"), "w", encoding="utf-8") as f:
        f.write("id\tlabel\texamples\tpercentage\n")
        for lab, idx in lab2id.items():
            n   = stats_lab.get(idx, 0)
            pct = 0 if total == 0 else 100 * n / total
            f.write(f"{idx}\t{lab}\t{n}\t{pct:.2f}%\n")

    with open(os.path.join(out_dir, "dataset_stats.txt"), "w", encoding="utf-8") as f:
        f.write("split\tflows\n")
        for s, n in stats_split:
            f.write(f"{s}\t{n}\n")

    report_path = os.path.join(out_dir, "dataset_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DATASET REPORT\n")
        f.write(f"Generated  : {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Source CSV : {os.path.abspath(csv_file)}\n")
        f.write(f"Keep [PKTSEP]: {'yes' if keep_pkt else 'no'}\n")
        f.write(f"Label column : {label_col}\n")
        f.write(f"Output dir  : {os.path.abspath(out_dir)}\n")
        f.write(f"Max flows (train): {max_flows if max_flows is not None else 'no limit'}\n\n")

        f.write("== FLOWS PER SPLIT ==\n")
        for s, n in stats_split:
            f.write(f"- {s:5s}: {n}\n")
        f.write("\n")

        f.write("== LABEL MAPPING (GLOBAL) ==\n")
        f.write("id\tlabel\texamples\tpercentage\n")
        for lab, idx in lab2id.items():
            n = stats_lab.get(idx, 0)
            pct = 0 if total == 0 else 100 * n / total
            f.write(f"{idx}\t{lab}\t{n}\t{pct:.2f}%\n")
        f.write("\n")

        for split in ("train", "dev", "test"):
            if split not in per_split_counts:
                continue
            counts = per_split_counts[split]
            n = int(counts.sum())
            present = set(counts.index.tolist())
            all_ids = set(range(len(labels)))
            missing = sorted(all_ids - present)
            maj_id = int(counts.idxmax()) if n else -1
            maj_ct = int(counts.max()) if n else 0
            maj_pct = (100.0 * maj_ct / n) if n else 0.0
            min_ct = int(counts[counts > 0].min()) if (counts > 0).any() else 0
            ir = (maj_ct / max(1, min_ct)) if n else 0.0

            f.write(f"== SPLIT: {split.upper()} ==\n")
            f.write(f"flows={n} | classes_present={len(present)}/{len(all_ids)} "
                    f"({0 if not all_ids else 100.0*len(present)/len(all_ids):.2f}%) | "
                    f"majority_id={maj_id if n else '-'} "
                    f"({id2lab.get(maj_id,'') if n else ''})={maj_pct:.2f}% | "
                    f"imbalance_ratio={ir:.2f}\n")
            if missing:
                f.write(f"missing_ids: {','.join(map(str, missing))}\n")

            f.write("id\tlabel\tcount\tpercent\n")
            for cid, ct in counts.sort_index().items():
                pct = 0 if n == 0 else 100.0 * ct / n
                f.write(f"{cid}\t{id2lab[cid]}\t{int(ct)}\t{pct:.2f}%\n")
            f.write("\n")

            lens = per_split_lengths[split]
            if len(lens):
                p50 = float(lens.quantile(0.5))
                p90 = float(lens.quantile(0.9))
                f.write("length_stats: ")
                f.write(f"min={int(lens.min())} p50={int(p50)} p90={int(p90)} "
                        f"max={int(lens.max())} mean={float(lens.mean()):.2f} "
                        f"std={float(lens.std(ddof=0)):.2f}\n\n")

    print("Hecho →", out_dir)
    print("Informe fusionado:", report_path)


if __name__ == "__main__":
    random.seed()
    pd.options.mode.chained_assignment = None
    main()
