import json, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
metrics_path = OUT / "metrics_ml_lr.json"
preds_path = OUT / "predictions_ml_lr.csv"

# --- Load data ---
metrics = json.loads(metrics_path.read_text())
df = pd.read_csv(preds_path)
y_true = df["y_true"]
y_score = df["y_score"]

# --- Style helper ---
def tidy(ax):
    ax.grid(True, alpha=0.2)
    for s in ax.spines.values():
        s.set_alpha(0.3)

# ======================================================
# 1️⃣ ROC & Precision–Recall Curves
# ======================================================
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--", lw=1)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
tidy(plt.gca())

plt.subplot(1, 2, 2)
plt.plot(rec, prec, label=f"AP = {ap:.4f}")
plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
tidy(plt.gca())

plt.tight_layout()
plt.savefig(OUT / "figure_roc_pr_curves.png", dpi=220)
print("✅ Saved →", OUT / "figure_roc_pr_curves.png")

# ======================================================
# 2️⃣ Metric Summary Bar Chart
# ======================================================
plt.figure(figsize=(7, 4))
labels = list(metrics.keys())
values = list(metrics.values())
plt.bar(labels, values, color="tab:blue", alpha=0.7)
plt.title("Model Performance Summary")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
tidy(plt.gca())
plt.tight_layout()
plt.savefig(OUT / "figure_metrics_summary.png", dpi=220)
print("✅ Saved →", OUT / "figure_metrics_summary.png")

# ======================================================
# 3️⃣ Score Distribution
# ======================================================
plt.figure(figsize=(6, 4))
plt.hist(y_score[y_true==1], bins=30, alpha=0.6, label="True Links (1)")
plt.hist(y_score[y_true==0], bins=30, alpha=0.6, label="Non-Links (0)")
plt.title("Score Distribution by Class")
plt.xlabel("Predicted Score")
plt.ylabel("Frequency")
plt.legend()
tidy(plt.gca())
plt.tight_layout()
plt.savefig(OUT / "figure_score_distribution.png", dpi=220)
print("✅ Saved →", OUT / "figure_score_distribution.png")
