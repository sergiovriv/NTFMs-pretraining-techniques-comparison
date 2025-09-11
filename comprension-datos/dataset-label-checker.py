import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/"
OUT_DIR  = "./dataset_qc_reports/"

os.makedirs(OUT_DIR, exist_ok=True)

for subdir in sorted(os.listdir(BASE_DIR)):
    path = os.path.join(BASE_DIR, subdir)
    if not os.path.isdir(path):
        continue

    tsvs = [
        fn for fn in glob.glob(os.path.join(path, "*_dataset.tsv"))
        if "nolabel" not in os.path.basename(fn)
    ]
    if not tsvs:
        print(f">>> No hay ficheros etiquetados en {subdir}, saltando.")
        continue

    dfs = []
    for fn in tsvs:
        try:
            df = pd.read_csv(fn, sep="\t", usecols=["label"])
        except ValueError:
            print(f"   – Salto {os.path.basename(fn)} (no tiene columna 'label').")
            continue
        dfs.append(df)

    if not dfs:
        print(f">>> Tras filtrar, no quedan datos en {subdir}, salto.")
        continue

    full = pd.concat(dfs, ignore_index=True)

    counts   = full["label"].value_counts().sort_index()
    total    = len(full)
    distinct = counts.size

    # informe TXT
    report_path = os.path.join(OUT_DIR, f"{subdir}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Dataset: {subdir}\n")
        f.write(f"Total filas (train+dev): {total}\n")
        f.write(f"Clases distintas:      {distinct}\n\n")
        f.write("Etiqueta\tRecuento\tPorcentaje\n")
        for lbl, cnt in counts.items():
            pct = cnt / total * 100
            f.write(f"{lbl}\t{cnt}\t{pct:.2f}%\n")

    plt.figure(figsize=(8,4))
    counts.plot.bar()
    plt.title(f"Distribución de etiquetas — {subdir}")
    plt.xlabel("Etiqueta")
    plt.ylabel("Recuento")
    plt.tight_layout()
    img_path = os.path.join(OUT_DIR, f"{subdir}_hist.png")
    plt.savefig(img_path)
    plt.close()

    print(f" Informe y gráfico generados para {subdir}")

print("Fin de la verificación.")
