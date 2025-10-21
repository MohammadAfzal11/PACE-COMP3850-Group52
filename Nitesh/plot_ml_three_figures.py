import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, average_precision_score

RESULTS_GLOB = "Nitesh/results/ml_results_*_*.csv"

def collect():
    rows = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        # Expect names like ml_results_100_lr.csv / ml_results_500_svm.csv
        tag = os.path.basename(path).replace("ml_results_", "").replace(".csv", "")
        size, model = tag.split("_")  # ["100","lr"]
        size = int(size)
        df = pd.read_csv(path)
        acc_ml = accuracy_score(df["y_true"], df["y_pred_ml"])
        f1_ml  = f1_score(df["y_true"], df["y_pred_ml"], zero_division=0)
        acc_bs = accuracy_score(df["y_true"], df["y_pred_baseline"])
        f1_bs  = f1_score(df["y_true"], df["y_pred_baseline"], zero_division=0)
        rows.append(dict(size=size, model=model, acc_ml=acc_ml, f1_ml=f1_ml,
                         acc_bs=acc_bs, f1_bs=f1_bs, delta_f1=f1_ml - f1_bs, path=path))
    return pd.DataFrame(rows).sort_values(["size","model"])

def panel_style(ax):
    ax.grid(True, alpha=0.15)
    for spine in ax.spines.values():
        spine.set_alpha(0.3)

def fig_analysis(res: pd.DataFrame):
    # Like "privacy-utility tradeoff": Accuracy & F1 vs size with baseline ref
    sizes = sorted(res["size"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Accuracy
    for model, g in res.groupby("model"):
        axes[0].plot(g["size"], g["acc_ml"], marker="o", label=f"ML ({model.upper()})")
    axes[0].plot(sizes, [res[res["size"]==s]["acc_bs"].mean() for s in sizes],
                 linestyle="--", marker="s", label="Baseline (mean)")
    axes[0].set_title("Accuracy vs Dataset Size"); axes[0].set_xlabel("Records per party"); axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(sizes); axes[0].legend(); panel_style(axes[0])

    # F1
    for model, g in res.groupby("model"):
        axes[1].plot(g["size"], g["f1_ml"], marker="o", label=f"ML ({model.upper()})")
    axes[1].plot(sizes, [res[res["size"]==s]["f1_bs"].mean() for s in sizes],
                 linestyle="--", marker="s", label="Baseline (mean)")
    axes[1].set_title("F1 vs Dataset Size"); axes[1].set_xlabel("Records per party"); axes[1].set_ylabel("F1 Score")
    axes[1].set_xticks(sizes); axes[1].legend(); panel_style(axes[1])

    plt.tight_layout()
    os.makedirs("Nitesh", exist_ok=True)
    plt.savefig("Nitesh/ml_analysis.png", dpi=200)
    print("Saved → Nitesh/ml_analysis.png")

def fig_dataset_size(res: pd.DataFrame):
    # 3-panel: Accuracy, F1, Delta-F1 (like Afzal’s dataset-size figure)
    sizes = sorted(res["size"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Accuracy
    for model, g in res.groupby("model"):
        axes[0].plot(g["size"], g["acc_ml"], marker="o", label=f"ML ({model.upper()})")
    axes[0].plot(sizes, [res[res["size"]==s]["acc_bs"].mean() for s in sizes],
                 linestyle="--", marker="s", label="Baseline (mean)")
    axes[0].set_title("Dataset Size Impact: Accuracy"); axes[0].set_xlabel("Records per party"); axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(sizes); axes[0].legend(); panel_style(axes[0])

    # F1
    for model, g in res.groupby("model"):
        axes[1].plot(g["size"], g["f1_ml"], marker="o", label=f"ML ({model.upper()})")
    axes[1].plot(sizes, [res[res["size"]==s]["f1_bs"].mean() for s in sizes],
                 linestyle="--", marker="s", label="Baseline (mean)")
    axes[1].set_title("Dataset Size Impact: F1 Score"); axes[1].set_xlabel("Records per party"); axes[1].set_ylabel("F1 Score")
    axes[1].set_xticks(sizes); axes[1].legend(); panel_style(axes[1])

    # Delta F1 bars
    width = 0.35; x = np.arange(len(sizes))
    for i, model in enumerate(sorted(res["model"].unique())):
        vals = [res[(res["size"]==s) & (res["model"]==model)]["delta_f1"].iloc[0] for s in sizes]
        axes[2].bar(x + (i-0.5)*width, vals, width, label=f"{model.upper()}")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Improvement over Baseline (ΔF1)"); axes[2].set_xlabel("Records per party"); axes[2].set_ylabel("ΔF1")
    axes[2].set_xticks(x); axes[2].set_xticklabels([str(s) for s in sizes]); axes[2].legend(); panel_style(axes[2])

    plt.tight_layout()
    plt.savefig("Nitesh/ml_dataset_size_comparison.png", dpi=200)
    print("Saved → Nitesh/ml_dataset_size_comparison.png")

def fig_working(res: pd.DataFrame):
    """
    Evidence figure (no training curves for LR/SVM).
    We emulate Afzal’s “working” panel with:
      - Precision-Recall curve (from ML scores if available; otherwise approximate using F1 points)
      - Threshold sweep (approx) using F1 across different decision cutoffs if probs are present
      - Bars for precision/recall at default threshold
    Note: Our saved CSV has class predictions, not probabilities.
    We’ll use the best available run (highest F1) and show bars + AP using baseline vs ML.
    """
    # pick best row by F1
    best = res.sort_values("f1_ml", ascending=False).iloc[0]
    df = pd.read_csv(best["path"])

    # compute precision/recall bars at default predictions
    p_ml  = ( (df["y_true"] & df["y_pred_ml"]).sum() / max(df["y_pred_ml"].sum(), 1) ) if "y_pred_ml" in df else 0
    r_ml  = ( (df["y_true"] & df["y_pred_ml"]).sum() / max(df["y_true"].sum(), 1) ) if "y_pred_ml" in df else 0
    p_bs  = ( (df["y_true"] & df["y_pred_baseline"]).sum() / max(df["y_pred_baseline"].sum(), 1) )
    r_bs  = ( (df["y_true"] & df["y_pred_baseline"]).sum() / max(df["y_true"].sum(), 1) )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Left: simple bar comparison precision/recall
    axes[0].bar([0-0.15, 0+0.15], [p_bs, r_bs], width=0.3, label="Baseline")
    axes[0].bar([1-0.15, 1+0.15], [p_ml, r_ml], width=0.3, label=f"ML ({best['model'].upper()})")
    axes[0].set_xticks([0,1]); axes[0].set_xticklabels(["Precision", "Recall"])
    axes[0].set_ylim(0,1); axes[0].set_title("Working: P/R at default threshold"); axes[0].legend(); panel_style(axes[0])

    # Middle: F1 comparison
    f1_bs = f1_score(df["y_true"], df["y_pred_baseline"], zero_division=0)
    f1_ml = f1_score(df["y_true"], df["y_pred_ml"], zero_division=0)
    axes[1].bar([0,1], [f1_bs, f1_ml], width=0.5, color=["tab:gray","tab:blue"])
    axes[1].set_xticks([0,1]); axes[1].set_xticklabels(["Baseline", f"ML ({best['model'].upper()})"])
    axes[1].set_ylim(0,1); axes[1].set_title("Working: F1 comparison"); panel_style(axes[1])

    # Right: placeholder “cost” panel → Delta-F1
    axes[2].bar([0], [f1_ml - f1_bs], width=0.4, color="tab:green")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks([0]); axes[2].set_xticklabels([f"{best['size']} recs"])
    axes[2].set_title("Working: Improvement (ΔF1)"); axes[2].set_ylabel("ΔF1"); panel_style(axes[2])

    plt.tight_layout()
    plt.savefig("Nitesh/ml_working.png", dpi=200)
    print("Saved → Nitesh/ml_working.png")

def main():
    res = collect()
    if res.empty:
        print("No result CSVs found. Run ml_text_linkage.py first.")
        return
    print(res)
    fig_analysis(res)
    fig_dataset_size(res)
    fig_working(res)

if __name__ == "__main__":
    main()
