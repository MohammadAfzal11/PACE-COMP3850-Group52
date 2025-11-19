# -*- coding: utf-8 -*-
# logistic_regression_evaluation.py
# Click ▶️ in VS Code to run with no args. It will auto-find the model & test CSV.

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    average_precision_score,  # AUCPR
    confusion_matrix,
)\


# --- unpickle shim: expose TextEmbedder under __main__ for old pickles ---
try:
    # try importing from your trainer module
    from Fardin.logistic_regression_linkage import TextEmbedder as _TE
except Exception:
    try:
        from logistic_regression_linkage import TextEmbedder as _TE  # fallback if run inside Fardin/
    except Exception:
        class _TE:  # last-resort stub; state will be restored from pickle
            pass
# make it visible as __main__.TextEmbedder
TextEmbedder = _TE
# --- end shim ---

# ---------- keep preprocessing & features IDENTICAL to training ----------
def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.lower()
    s = " ".join(s.split())
    return s

def pair_features(u: np.ndarray, v: np.ndarray, len1: np.ndarray, len2: np.ndarray) -> np.ndarray:
    abs_diff = np.abs(u - v)
    hadamard = u * v
    u_norm = np.linalg.norm(u, axis=1) + 1e-12
    v_norm = np.linalg.norm(v, axis=1) + 1e-12
    cos = (u * v).sum(axis=1) / (u_norm * v_norm)
    cos = cos.reshape(-1, 1)
    len1 = len1.reshape(-1, 1)
    len2 = len2.reshape(-1, 1)
    len_diff = np.abs(len1 - len2)
    len_ratio = (np.minimum(len1, len2) / np.maximum(len1, len2)).astype(np.float32)
    return np.hstack([abs_diff, hadamard, cos, len1, len2, len_diff, len_ratio])

# ---------- evaluator ----------
def evaluate(model: Dict, df: pd.DataFrame, text1: str, text2: str, label: str, threshold: float | None = None):
    embed1 = model["embed1"]
    embed2 = model["embed2"]
    clf    = model["clf"]
    tau    = float(model.get("tau", 0.5)) if threshold is None else float(threshold)

    t1 = df[text1].astype(str).values
    t2 = df[text2].astype(str).values
    y_true = df[label].astype(int).values
    len1 = np.array([len(norm_text(x)) for x in t1], dtype=np.float32)
    len2 = np.array([len(norm_text(x)) for x in t2], dtype=np.float32)

    u = embed1.transform(t1)
    v = embed2.transform(t2)
    X = pair_features(u, v, len1, len2)

    scores = clf.predict_proba(X)[:, 1]
    y_pred = (scores >= tau).astype(int)

    pr_auc = float(average_precision_score(y_true, scores))     # <-- AUCPR headline
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    roc = float(roc_auc_score(y_true, scores))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": tau,
        "pr_auc": pr_auc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": acc,
        "roc_auc": roc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n": int(len(df)),
    }
    return metrics, scores

# ---------- auto-discovery so ▶️ works ----------
def _autopaths():
    here = Path(__file__).resolve().parent
    model = here / "ml_linkage_baseline.joblib"
    # try both locations for the test CSV
    candidates = [
        here / "target_test.csv",
        here.parent / "csv_files" / "target_test.csv",
    ]
    for c in candidates:
        if c.exists():
            return model, c
    # fall back to the last candidate (so argparse will kick in)
    return model, candidates[-1]

if __name__ == "__main__":
    # try auto-paths first (so you can just click ▶️)
    model_path, test_csv = _autopaths()
    if not model_path.exists() or not test_csv.exists():
        # fall back to CLI args if files aren't where we expect
        import argparse
        ap = argparse.ArgumentParser(description="Evaluate saved ML linkage model (prints AUCPR)")
        ap.add_argument("--model_path", type=str, required=True)
        ap.add_argument("--test_csv",   type=str, required=True)
        ap.add_argument("--text1", type=str, default="text1")
        ap.add_argument("--text2", type=str, default="text2")
        ap.add_argument("--label", type=str, default="label")
        ap.add_argument("--threshold", type=float, default=None, help="Optional override of saved τ")
        ap.add_argument("--preds_out", type=str, default=None, help="Optional CSV of per-row scores & preds")
        ap.add_argument("--report_json", type=str, default=None, help="Optional JSON metrics dump")
        args = ap.parse_args()
        model_path = Path(args.model_path)
        test_csv   = Path(args.test_csv)
        text1, text2, label = args.text1, args.text2, args.label
        threshold = args.threshold
        preds_out = args.preds_out
        report_json = args.report_json
    else:
        # defaults for one-click run
        text1, text2, label = "text1", "text2", "label"
        threshold = None
        preds_out = None
        report_json = None

    # load + eval
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_csv)
    metrics, scores = evaluate(model, test_df, text1, text2, label, threshold)

    # print AUCPR plainly, then full JSON
    print(f"AUCPR (PR-AUC): {metrics['pr_auc']:.6f}")
    print(json.dumps(metrics, indent=2))

    # optional outputs
    if preds_out:
        out = test_df.copy()
        out["score"] = scores
        out["pred"]  = (scores >= metrics["threshold"]).astype(int)
        out.to_csv(preds_out, index=False)

    if report_json:
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
