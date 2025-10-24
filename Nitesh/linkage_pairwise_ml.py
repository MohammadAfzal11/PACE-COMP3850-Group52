#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid ML + Bloom Filter (with optional DP) for Pairwise Linkage
----------------------------------------------------------------
Covers everything your brief asks:
- Unstructured text features: word n-gram cosine (hashing), optional char n-gram cosine,
  token Jaccard, length diff, digit overlap
- Structured features: numeric (diff/|diff|/ratio/min/max/mean), categorical (exact + Jaccard),
  multi-valued categoricals via --multi-cat base:separator (e.g., codes:; tags:|)
- Bloom Filter similarity from your src/BF.py (binary BF Dice) via --use-bf
- Optional Differential Privacy (Laplace output noise) for BF score (--dp-bf) and text cosines (--dp-text)
- Validation split to pick threshold (no test leakage)
- Low-RAM switches: --n-features, --no-char

"""

from __future__ import annotations

import argparse, os, json, math, sys, random, platform
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

# ---------- Optional BF (your src/BF.py) ----------
BF_DICE_AVAILABLE = False
try:
    # Your file defines class BF with helpers; we’ll call its methods
    from BF import BF as BFClass
    BF_DICE_AVAILABLE = True
except Exception:
    BF_DICE_AVAILABLE = False

# ---------- Globals ----------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ============================ Helpers ============================
def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)): return ""
    return str(x)

def _normalize_text(s: str) -> str:
    s = _safe_str(s).lower()
    out, prev = [], False
    for ch in s:
        if ch.isalnum():
            out.append(ch); prev = False
        else:
            if not prev: out.append(" "); prev = True
    return "".join(out).strip()

def tokenize_words(s: str) -> List[str]:
    return [t for t in _normalize_text(s).split() if t]

def jaccard_tokens(a: str, b: str) -> float:
    A, B = set(tokenize_words(a)), set(tokenize_words(b))
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / (len(A | B) + 1e-12)

def jaccard_multivalue(a: str, b: str, sep: str) -> float:
    A = {t.strip().lower() for t in _safe_str(a).split(sep) if t.strip()}
    B = {t.strip().lower() for t in _safe_str(b).split(sep) if t.strip()}
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / (len(A | B) + 1e-12)

def _digits_only(s: str) -> str:
    return "".join(ch for ch in _safe_str(s) if ch.isdigit())

def cosine_sim_sparse(X: sparse.csr_matrix, Y: sparse.csr_matrix) -> np.ndarray:
    # Row-wise cosine for CSR matrices
    def _row_norm(m: sparse.csr_matrix):
        rn = np.sqrt(m.multiply(m).sum(axis=1)).A1 + 1e-12
        return sparse.diags(1.0 / rn) @ m
    Xn, Yn = _row_norm(X), _row_norm(Y)
    return (Xn.multiply(Yn)).sum(axis=1).A1

# ---------- BF Dice using your BF.py ----------
def bf_dice_series(s1: pd.Series, s2: pd.Series, q: int = 2, m: int = 2048, k: int = 10) -> Optional[np.ndarray]:
    """Compute Dice similarity between Bloom filters built from q-grams using BFClass."""
    if not BF_DICE_AVAILABLE: return None
    sims = np.zeros(len(s1), dtype=float)
    bf = BFClass(bf_len=m, bf_num_hash_func=k, q=q)
    for i, (a, b) in enumerate(zip(s1.fillna("").astype(str), s2.fillna("").astype(str))):
        set1 = bf.convert_str_val_to_set(a)
        set2 = bf.convert_str_val_to_set(b)
        bf1  = bf.set_to_bloom_filter(set1)
        bf2  = bf.set_to_bloom_filter(set2)
        sims[i] = bf.calc_bf_sim(bf1, bf2)  # Dice similarity
    return sims


# ====================== Structured helpers ======================
def infer_paired_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect paired columns like base1/base2 (excluding uid/text bases).
    Returns {'num': [...], 'cat': [...]} base names (without '1'/'2').
    """
    bases = {}
    for c in df.columns:
        if c.endswith("1") and (c[:-1] + "2") in df.columns:
            base = c[:-1]
            if base.lower() in {"uid", "text"}:  # ignore the core text/uid
                continue
            bases[base] = (c, c[:-1] + "2")

    typed = {"num": [], "cat": []}
    for base in sorted(bases):
        s1, s2 = df[base + "1"], df[base + "2"]
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            typed["num"].append(base)
        else:
            typed["cat"].append(base)
    return typed


# ========================= Feature builder =========================
def build_features_pairwise(
    df: pd.DataFrame,
    n_features: int = 2**16,
    char_ngrams: Optional[Tuple[int, int]] = (2, 5),
    multi_cat_separators: Optional[Dict[str, str]] = None,
    use_bf: bool = False,
    dp_text_scale: float = 0.0,
    dp_bf_scale: float = 0.0
) -> Tuple[pd.DataFrame, List[str]]:

    need = {"uid1", "text1", "uid2", "text2"}
    if not need.issubset(df.columns):
        raise ValueError(f"Expected columns: {sorted(list(need))} (plus 'label').")

    feats, used = [], []

    # ---------- Text features ----------
    t1 = df["text1"].fillna("").astype(str)
    t2 = df["text2"].fillna("").astype(str)

    # Word hashing cosine
    wv = HashingVectorizer(ngram_range=(1, 2), alternate_sign=False, norm=None,
                           lowercase=True, analyzer="word", n_features=n_features)
    Xw1, Xw2 = wv.transform(t1), wv.transform(t2)
    word_cos = cosine_sim_sparse(Xw1, Xw2)

    # Optional char hashing cosine
    char_cos = np.zeros(len(t1))
    if char_ngrams is not None:
        cv = HashingVectorizer(ngram_range=char_ngrams, alternate_sign=False, norm=None,
                               lowercase=True, analyzer="char", n_features=n_features)
        Xc1, Xc2 = cv.transform(t1), cv.transform(t2)
        char_cos = cosine_sim_sparse(Xc1, Xc2)

    # Token Jaccard, length diff, digit overlap
    jac = [jaccard_tokens(a, b) for a, b in zip(t1.tolist(), t2.tolist())]
    len_diff = np.abs(t1.str.len() - t2.str.len()).astype(float).values
    digit_overlap = [1.0 if (set(_digits_only(a)) & set(_digits_only(b))) else 0.0 for a, b in zip(t1.tolist(), t2.tolist())]

    # Optional DP noise on cosine scores (output perturbation)
    if dp_text_scale and dp_text_scale > 0:
        word_cos = np.clip(word_cos + np.random.laplace(0.0, dp_text_scale, size=word_cos.shape), 0.0, 1.0)
        char_cos = np.clip(char_cos + np.random.laplace(0.0, dp_text_scale, size=char_cos.shape), 0.0, 1.0)

    tdf = pd.DataFrame({
        "text_word_cos": word_cos,
        "text_char_cos": char_cos,
        "text_jaccard": jac,
        "text_len_diff": len_diff,
        "text_digit_overlap": digit_overlap
    })
    feats.append(tdf); used += list(tdf.columns)

    # ---------- Bloom Filter Dice (from BF.py) ----------
    if use_bf and BF_DICE_AVAILABLE:
        bf_dice = bf_dice_series(t1, t2, q=2, m=2048, k=10)
        if dp_bf_scale and dp_bf_scale > 0:
            bf_dice = np.clip(bf_dice + np.random.laplace(0.0, dp_bf_scale, size=bf_dice.shape), 0.0, 1.0)
        feats.append(pd.DataFrame({"text_bf_dice": bf_dice}))
        used.append("text_bf_dice")

    # ---------- Structured features ----------
    typed = infer_paired_columns(df)

    # Numeric: diff, abs diff, ratio, min/max/mean
    for base in typed["num"]:
        a = pd.to_numeric(df[base + "1"], errors="coerce")
        b = pd.to_numeric(df[base + "2"], errors="coerce")
        diff = (a - b).astype(float); adiff = diff.abs()
        ratio = (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        mini = pd.concat([a, b], axis=1).min(axis=1).fillna(0.0)
        maxi = pd.concat([a, b], axis=1).max(axis=1).fillna(0.0)
        meanv= pd.concat([a, b], axis=1).mean(axis=1).fillna(0.0)
        nd = pd.DataFrame({
            f"{base}_num_diff":  diff.fillna(0.0).values,
            f"{base}_num_adiff": adiff.fillna(0.0).values,
            f"{base}_num_ratio": ratio.values,
            f"{base}_num_min":   mini.values,
            f"{base}_num_max":   maxi.values,
            f"{base}_num_mean":  meanv.values
        })
        feats.append(nd); used += list(nd.columns)

    # Categorical: exact + Jaccard (or multivalue Jaccard if specified)
    for base in typed["cat"]:
        a = df[base + "1"].fillna("").astype(str)
        b = df[base + "2"].fillna("").astype(str)
        exact = (a.str.lower() == b.str.lower()).astype(int).values
        sep = multi_cat_separators.get(base) if multi_cat_separators else None
        if sep:
            j = [jaccard_multivalue(x, y, sep) for x, y in zip(a.tolist(), b.tolist())]
        else:
            j = [jaccard_tokens(x, y) for x, y in zip(a.tolist(), b.tolist())]
        cd = pd.DataFrame({f"{base}_cat_exact": exact, f"{base}_cat_jacc": np.array(j, float)})
        feats.append(cd); used += list(cd.columns)

    # UID exact (sanity)
    uid_exact = (df["uid1"].astype(str) == df["uid2"].astype(str)).astype(int)
    feats.append(pd.DataFrame({"uid_exact": uid_exact.values})); used.append("uid_exact")

    X = pd.concat(feats, axis=1)
    return X, used


# ============================ Models & Eval ============================
def build_model(kind: str = "lr"):
    k = kind.lower()
    if k in {"lr", "logistic"}:
        return LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
    if k == "svm":
        base = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
        return CalibratedClassifierCV(base, method="sigmoid", cv=5)
    if k == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE)
    if k in {"gb", "gbrt"}:
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    raise ValueError("Unknown model: " + kind)

def evaluate_threshold_free(y_true, y_score):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score))
    }

def save_run_info(output_dir: str, cfg: dict, metrics: dict):
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "random_state": RANDOM_STATE,
        "config": cfg,
        "metrics_snapshot": metrics
    }
    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)


# =============================== Pipeline ===============================
def run(train_path: str, test_path: str, model_name: str, output_dir: str,
        n_features: int, no_char: bool, use_bf: bool, dp_bf_scale: float,
        dp_text_scale: float, multi_cat: Optional[List[str]], val_size: float):

    os.makedirs(output_dir, exist_ok=True)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    if "label" not in train.columns or "label" not in test.columns:
        raise ValueError("Both train and test must contain a 'label' column.")

    # Parse multi-cat map like ["codes:;", "tags:|"]
    multi_map = {}
    if multi_cat:
        for item in multi_cat:
            if ":" in item:
                base, sep = item.split(":", 1)
                multi_map[base.strip()] = sep

    # -------- Build TRAIN features; split for validation --------
    X_all, _ = build_features_pairwise(
        train,
        n_features=n_features,
        char_ngrams=None if no_char else (2, 5),
        multi_cat_separators=multi_map,
        use_bf=(use_bf and BF_DICE_AVAILABLE),
        dp_text_scale=dp_text_scale,
        dp_bf_scale=dp_bf_scale
    )
    y_all = train["label"].astype(int).values

    Xtr, Xval, ytr, yval = train_test_split(
        X_all, y_all, test_size=val_size, stratify=y_all, random_state=RANDOM_STATE
    )

    model = build_model(model_name)
    model.fit(Xtr, ytr)

    # -------- Choose threshold on VALIDATION --------
    if hasattr(model, "predict_proba"):
        val_score = model.predict_proba(Xval)[:, 1]
    else:
        z = model.decision_function(Xval)
        val_score = 1.0 / (1.0 + np.exp(-z))
    prec, rec, thr = precision_recall_curve(yval, val_score)
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    thr_opt = float(thr[np.nanargmax(f1s)]) if len(thr) else 0.5

    # -------- Build TEST features and evaluate --------
    Xte, _ = build_features_pairwise(
        test,
        n_features=n_features,
        char_ngrams=None if no_char else (2, 5),
        multi_cat_separators=multi_map,
        use_bf=(use_bf and BF_DICE_AVAILABLE),
        dp_text_scale=dp_text_scale,
        dp_bf_scale=dp_bf_scale
    )
    yte = test["label"].astype(int).values

    if hasattr(model, "predict_proba"):
        te_score = model.predict_proba(Xte)[:, 1]
    else:
        z = model.decision_function(Xte)
        te_score = 1.0 / (1.0 + np.exp(-z))

    y_pred = (te_score >= thr_opt).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, y_pred).ravel().tolist()

    metrics = evaluate_threshold_free(yte, te_score)
    metrics.update({
        "threshold_from_validation": thr_opt,
        "accuracy": float(accuracy_score(yte, y_pred)),
        "precision": float(precision_score(yte, y_pred, zero_division=0)),
        "recall": float(recall_score(yte, y_pred, zero_division=0)),
        "f1": float(f1_score(yte, y_pred, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    })

    # -------- Save artifacts --------
    out_pred = os.path.join(output_dir, f"predictions_ml_{model_name}.csv")
    pd.DataFrame({
        "uid1": test.get("uid1", pd.Series(index=test.index)),
        "uid2": test.get("uid2", pd.Series(index=test.index)),
        "text1": test.get("text1", pd.Series(index=test.index)),
        "text2": test.get("text2", pd.Series(index=test.index)),
        "y_true": yte, "y_pred": y_pred, "y_score": te_score
    }).to_csv(out_pred, index=False)

    out_metrics = os.path.join(output_dir, f"metrics_ml_{model_name}.json")
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # Error analysis
    errs = pd.DataFrame({
        "uid1": test.get("uid1", pd.Series(index=test.index)),
        "uid2": test.get("uid2", pd.Series(index=test.index)),
        "text1": test.get("text1", pd.Series(index=test.index)),
        "text2": test.get("text2", pd.Series(index=test.index)),
        "score": te_score, "y_true": yte, "y_pred": y_pred
    })
    fp_df = errs[(errs.y_true == 0) & (errs.y_pred == 1)].sort_values("score", ascending=False).head(200)
    fn_df = errs[(errs.y_true == 1) & (errs.y_pred == 0)].sort_values("score", ascending=True).head(200)
    fp_df.to_csv(os.path.join(output_dir, "top_false_positives.csv"), index=False)
    fn_df.to_csv(os.path.join(output_dir, "top_false_negatives.csv"), index=False)

    save_run_info(output_dir, {
        "model": model_name,
        "n_features": n_features,
        "no_char": no_char,
        "use_bf": bool(use_bf and BF_DICE_AVAILABLE),
        "dp_bf_scale": dp_bf_scale,
        "dp_text_scale": dp_text_scale,
        "val_size": val_size,
        "multi_cat": multi_cat
    }, metrics)

    print("\n=== Evaluation (Hybrid ML + BF; threshold from validation) ===")
    for k in ["threshold_from_validation","accuracy","precision","recall","f1","roc_auc","pr_auc","tn","fp","fn","tp"]:
        print(f"{k:>26}: {metrics.get(k)}")
    print(f"\nSaved predictions → {out_pred}")
    print(f"Saved metrics     → {out_metrics}")
    print("Saved errors      → top_false_positives.csv / top_false_negatives.csv")
    print("Saved run info    → run_info.json")


# ================================ CLI ================================
def parse_args():
    p = argparse.ArgumentParser(description="Hybrid ML + Bloom Filter (optional DP) for pairwise linkage.")
    p.add_argument("--train", required=True, help="Path to target_train.csv")
    p.add_argument("--test", required=True, help="Path to target_test.csv")
    p.add_argument("--model", default="lr", choices=["lr","svm","rf","gb"])
    p.add_argument("--output-dir", default="results")
    p.add_argument("--n-features", type=int, default=2**16, help="Hashing vector size (e.g., 65536)")
    p.add_argument("--no-char", action="store_true", help="Disable char n-grams (saves RAM)")
    p.add_argument("--use-bf", action="store_true", help="Add Bloom-filter Dice similarity using src/BF.py")
    p.add_argument("--dp-bf", type=float, default=0.0, help="Laplace scale for BF Dice (0 = no DP noise)")
    p.add_argument("--dp-text", type=float, default=0.0, help="Laplace scale for text cosines (0 = no DP noise)")
    p.add_argument("--multi-cat", nargs="*", default=None, help="Multi-valued categorical bases with separators, e.g. codes:; tags:|")
    p.add_argument("--val-size", type=float, default=0.15, help="Validation fraction for threshold selection")
    return p.parse_args()

def main():
    args = parse_args()

    # Safer defaults for low-RAM laptops (can be overridden by env)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    run(
        train_path=args.train,
        test_path=args.test,
        model_name=args.model,
        output_dir=args.output_dir,
        n_features=int(args.n_features),
        no_char=bool(args.no_char),
        use_bf=bool(args.use_bf),
        dp_bf_scale=float(args.dp_bf),
        dp_text_scale=float(args.dp_text),
        multi_cat=args.multi_cat,
        val_size=float(args.val_size)
    )

if __name__ == "__main__":
    main()
