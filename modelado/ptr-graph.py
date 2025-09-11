
# Usage:
# python3 plot_log.py --log path/to/logfile.log --outdir path/to/output --title "Pretraining Loss" --output_name loss_plot

import re
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def detect_type(lines):
    for line in lines:
        if 'loss_mlm' in line and 'loss_sp' in line:
            return "complex"
        elif 'loss' in line and 'acc:' in line:
            return "simple"
    return None

def process_complex(lines):
    data = []
    for line in lines:
        match = re.search(r"\s+(\d+)/\s+\d+\s+steps.*?loss\s+([\d.]+)\| loss_mlm: ([\d.]+)\| loss_sp: ([\d.]+)", line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            loss_mlm = float(match.group(3))
            loss_sp = float(match.group(4))
            data.append((step, loss, loss_mlm, loss_sp))
    return pd.DataFrame(data, columns=["step", "loss", "loss_mlm", "loss_sp"])

def process_simple(lines):
    data = []
    for line in lines:
        match = re.search(r"\s+(\d+)/\s+\d+\s+steps.*?loss\s+([\d.]+)\s+\|\s+acc:", line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            data.append((step, loss))
    return pd.DataFrame(data, columns=["step", "loss"])

def plot(df, type_, output_dir, title, output_name):
    plt.figure(figsize=(12, 6))

    window_size = 50
    if type_ == "complex":
        df["loss_avg"] = df["loss"].rolling(window=window_size, min_periods=1).mean()
        df["mlm_avg"] = df["loss_mlm"].rolling(window=window_size, min_periods=1).mean()
        df["sp_avg"] = df["loss_sp"].rolling(window=window_size, min_periods=1).mean()

        plt.plot(df["step"], df["loss_avg"], label="Mean Loss", color="black", linestyle="-", linewidth=1.5)
        plt.plot(df["step"], df["mlm_avg"], label="Loss MLM", color="black", linestyle="--", linewidth=1.0)
        plt.plot(df["step"], df["sp_avg"], label="Loss SP", color="black", linestyle=":", linewidth=1.0)

    else:
        df["loss_avg"] = df["loss"].rolling(window=window_size, min_periods=1).mean()
        plt.plot(df["step"], df["loss_avg"], label="Loss", color="black", linestyle="-", linewidth=1.5)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 520000)
    plt.ylim(0, 10.0)

    output_path = os.path.join(output_dir, f"{output_name}.svg")
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot pretraining loss from a log file.")
    parser.add_argument("--log", required=True, help="Path to log file (.log)")
    parser.add_argument("--outdir", required=True, help="Output directory for the plot")
    parser.add_argument("--title", required=True, help="Plot title")
    parser.add_argument("--output_name", required=True, help="Output file name (without extension)")

    args = parser.parse_args()

    with open(args.log, "r") as f:
        lines = f.readlines()

    type_ = detect_type(lines)
    if type_ == "complex":
        df = process_complex(lines)
    elif type_ == "simple":
        df = process_simple(lines)
    else:
        raise ValueError("Unrecognized log format.")

    plot(df, type_, args.outdir, args.title, args.output_name)

if __name__ == "__main__":
    main()
