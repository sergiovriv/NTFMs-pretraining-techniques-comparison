
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

def find_csvs(root: str) -> List[str]:
    csvs = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".csv"):
                csvs.append(os.path.join(r, fn))
    return sorted(csvs)

def stats_from_lengths(lengths: List[int]):
    if not lengths:
        return None
    n = len(lengths)
    lengths_sorted = sorted(lengths)
    min_v = lengths_sorted[0]
    max_v = lengths_sorted[-1]
    if n % 2 == 1:
        med = lengths_sorted[n // 2]
    else:
        med = 0.5 * (lengths_sorted[n // 2 - 1] + lengths_sorted[n // 2])
    mean = sum(lengths_sorted) / n
    return min_v, med, mean, max_v

def analyze_csv(path: str) -> Tuple[str, int, Dict[str, int], Tuple[int, float, float, int] or None]:
    flows = 0
    dist: Dict[str, int] = {}
    lengths: List[int] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            has_dt = "dataset_type" in (reader.fieldnames or [])
            has_text = "text" in (reader.fieldnames or [])
            for row in reader:
                flows += 1
                if has_dt:
                    raw = row.get("dataset_type", "")
                    key = (raw if raw is not None else "").strip()
                    if key == "":
                        key = "<EMPTY>"
                else:
                    key = "<NO_COLUMN>"
                dist[key] = dist.get(key, 0) + 1
                if has_text:
                    txt = row.get("text", "") or ""
                    lengths.append(len(txt.split()))
    except Exception as e:
        print(f"[WARN] No se pudo procesar {os.path.basename(path)}: {e}", file=sys.stderr)
        return (os.path.basename(path), 0, {}, None)
    stats = stats_from_lengths(lengths)
    return (os.path.basename(path), flows, dist, stats)

def main():
    ap = argparse.ArgumentParser(description="netbench-analysis estadístico multihilo")
    ap.add_argument("--input", required=True, help="Directorio raíz con CSVs de NetBench")
    ap.add_argument("--output", required=True, help="Ruta del informe .txt a generar")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="Hilos en paralelo")
    args = ap.parse_args()

    csv_files = find_csvs(args.input)
    if not csv_files:
        print("No se encontraron .csv en el directorio indicado.", file=sys.stderr)
        sys.exit(2)

    total = len(csv_files)
    print(f"Encontrados {total} CSV. Procesando con {args.workers} hilos...")

    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut2path = {ex.submit(analyze_csv, p): p for p in csv_files}
        for fut in as_completed(fut2path):
            res = fut.result()
            results.append(res)
            done += 1
            print(f"[{done}/{total}] {os.path.basename(fut2path[fut])} listo.", flush=True)

    lines = []
    for name, flows, dist, stats in sorted(results, key=lambda x: x[0].lower()):
        lines.append(f"# {name}")
        lines.append(f"Numero de flujos: {flows}")
        lines.append("Distribución por dataset_type:")
        if dist:
            for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"  {k}: {v}")
        else:
            lines.append("  <sin datos>")
        lines.append("Estadísticas de longitud de tokens:")
        if stats is None:
            lines.append("  <sin datos>")
        else:
            min_v, med, mean, max_v = stats
            lines.append(f"  min: {int(min_v)}")
            if isinstance(med, float) and not med.is_integer():
                lines.append(f"  mediana: {med:.2f}")
            else:
                lines.append(f"  mediana: {int(med)}")
            lines.append(f"  media: {mean:.2f}")
            lines.append(f"  max: {int(max_v)}")
        lines.append("")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Informe escrito en {args.output}")
    total_flows = sum(x[1] for x in results)
    print(f"Resumen: archivos={len(results)} | flujos totales={total_flows}")

if __name__ == "__main__":
    main()
