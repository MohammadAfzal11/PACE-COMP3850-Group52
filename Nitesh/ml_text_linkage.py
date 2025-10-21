#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-aware ML Linkage (single file)
- Text: q-grams + Counting Bloom Filter (CBF) + cosine similarity
- Numeric/Categorical: simple differences / equality flags
- Classifier: Logistic Regression (default) or SVM-RBF
- Baseline: simple threshold-style rule for comparison
- Outputs: printed metrics; optional predictions CSV and metrics JSON
"""

import argparse, json, math, hashlib, os
from typing import Dict, List, Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------------------
# Text normalization & CBF
# ---------------------------

def _normalize_text(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).lower()
    out, prev_space = [], False
    for ch in s:
        if "a" <= ch <= "z" or "0" <= ch <= "9":
            out.append(ch); prev_space = False
        else:
            if not prev_space:
                out.append(" "); prev_space = True
    return "".join(out).strip()

def _qgrams(s: str, q: int = 2) -> List[str]:
    s = _normalize_text(s)
    if not s:
        return []
    s = f"^{s}$"
    if len(s) < q:
        return [s]
    return [s[i:i+q] for i in range(len(s) - q + 1)]

def _hash_with_seed(token: str, seed: int) -> int:
    h = hashlib.blake2b(digest_size=8, person=seed.to_bytes(8, "little"))
    h.update(token.encode("utf-8"))
    return int.from_bytes(h.digest(), "little")

def encode_texts_cbf(texts: Iterable[str], m: int = 512, k: int = 3, q: int = 2, salt: str = "") -> np.ndarray:
    vec = np.zeros(m, dtype=np.int32)
    for t in texts:
        for g in _qgrams(t or "", q=q):
            tok = f"{salt}:{g}" if salt else g
            for i in range(k):
                vec[_hash_with_seed(tok, i) % m] += 1
    return vec

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = float(np.linalg.norm(a)); db = float(np.linalg.norm(b))
    if da == 0.0 and db == 0.0: return 1.0
    if da == 0.0 or db == 0.0:  return 0.0
    return float(np.dot(a, b) / (da * db))

# ---------------------------
# Schema, pairs, features
# ---------------------------

def auto_schema(A: pd.DataFrame, B: pd.DataFrame, idA: str, idB: str) -> Dict[str,str]:
    schema: Dict[str,str] = {}
    for c in set(A.columns) | set(B.columns):
        lc = str(c).lower()
        if lc in (idA.lower(), idB.lower()):
            continue
        if any(k in lc for k in ["postcode","zip","age","year","lat","lon","long"]):
            schema[c] = "numeric"
        elif any(k in lc for k in ["name","givenname","surname","address","addr","street","suburb","city","state","email","phone"]):
            schema[c] = "text"
        else:
            schema[c] = "categorical"
    return schema

def build_pairs(A: pd.DataFrame, B: pd.DataFrame,
                idA: str = "rec_id", idB: str = "recid",
                neg_per_pos: int = 3, seed: int = 42) -> pd.DataFrame:
    A_idx = A.set_index(idA); B_idx = B.set_index(idB)
    pos = sorted(set(A_idx.index) & set(B_idx.index))
    rng = np.random.default_rng(seed)
    pairs = [(i, i, 1) for i in pos]
    A_ids = np.array(A_idx.index)
    for j in B_idx.index:
        for _ in range(neg_per_pos):
            i = rng.choice(A_ids)
            if i == j: continue
            pairs.append((i, j, 0))
    return pd.DataFrame(pairs, columns=[idA, idB, "label"])

def assemble_features(A: pd.DataFrame, B: pd.DataFrame, pairs: pd.DataFrame,
                      schema: Dict[str,str],
                      idA: str = "rec_id", idB: str = "recid",
                      text_m: int = 512, text_k: int = 3, text_q: int = 2, salt: str = "mltext") -> pd.DataFrame:
    A_idx = A.set_index(idA); B_idx = B.set_index(idB)
    rows = []
    for _, r in pairs.iterrows():
        a = A_idx.loc[r[idA]]; b = B_idx.loc[r[idB]]
        feats: Dict[str, float] = {}

        # global text similarity across all text fields
        a_texts = [a[c] for c,t in schema.items() if t=="text" and c in a.index]
        b_texts = [b[c] for c,t in schema.items() if t=="text" and c in b.index]
        if a_texts or b_texts:
            va = encode_texts_cbf(a_texts, m=text_m, k=text_k, q=text_q, salt=salt)
            vb = encode_texts_cbf(b_texts, m=text_m, k=text_k, q=text_q, salt=salt)
            feats["text_cbf_cosine"] = cosine(va, vb)

        for col, typ in schema.items():
            if col not in a.index or col not in b.index: continue
            if typ == "numeric":
                try:
                    feats[f"num_absdiff::{col}"] = abs(float(a[col]) - float(b[col]))
                except Exception:
                    feats[f"num_absdiff::{col}"] = 1.0
            elif typ == "categorical":
                aa = _normalize_text(a[col]); bb = _normalize_text(b[col])
                feats[f"cat_eq::{col}"] = 1.0 if (aa!="" and aa == bb) else 0.0
            elif typ == "text":
                aa = _normalize_text(a[col]); bb = _normalize_text(b[col])
                feats[f"text_eq::{col}"] = 1.0 if (aa!="" and aa == bb) else 0.0

            if typ == "numeric":
                feats[f"num_log1p_diff::{col}"] = float(np.log1p(feats[f"num_absdiff::{col}"]))

        feats["label"] = int(r.get("label", 0))
        feats["idA"] = r[idA]; feats["idB"] = r[idB]
        rows.append(feats)

    return pd.DataFrame(rows).fillna(0.0)

# ---------------------------
# Baseline & ML training
# ---------------------------

def baseline_predict(feat_df: pd.DataFrame) -> np.ndarray:
    score = np.zeros(len(feat_df))
    for c in feat_df.columns:
        if c.startswith("text_eq::") or c.startswith("cat_eq::postcode") or c.startswith("cat_eq::zip"):
            score += feat_df[c].values
    return (score >= 1.0).astype(int)

def train_and_eval(F: pd.DataFrame, model: str = "lr") -> Tuple[dict, dict]:
    feat_cols = [c for c in F.columns if c not in ("label","idA","idB")]
    X = F[feat_cols].values
    y = F["label"].values.astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=11, stratify=y)

    if model == "lr":
        est = LogisticRegression(max_iter=1000)
    elif model == "svm":
        est = make_pipeline(StandardScaler(with_mean=False),
                            SVC(kernel="rbf", class_weight="balanced", C=2.0, gamma="scale"))
    else:
        raise ValueError("model must be 'lr' or 'svm'")

    est.fit(Xtr, ytr)
    yhat_ml = est.predict(Xte)

    F_te = pd.DataFrame(Xte, columns=feat_cols)
    yhat_bs = baseline_predict(F_te)

    p_ml, r_ml, f_ml, _ = precision_recall_fscore_support(yte, yhat_ml, average="binary", zero_division=0)
    p_bs, r_bs, f_bs, _ = precision_recall_fscore_support(yte, yhat_bs, average="binary", zero_division=0)

    return ({"precision":float(p_ml),"recall":float(r_ml),"f1":float(f_ml)},
            {"precision":float(p_bs),"recall":float(r_bs),"f1":float(f_bs)})

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Text-aware ML linkage using CBF features (single file).")
    ap.add_argument("--alice", required=True, help="Path to Alice CSV")
    ap.add_argument("--bob",   required=True, help="Path to Bob CSV")
    ap.add_argument("--idA", default="rec_id", help="Alice ID column (default: rec_id)")
    ap.add_argument("--idB", default="recid",  help="Bob ID column (default: recid)")
    ap.add_argument("--schema", help="Optional schema JSON: {col: 'text'|'numeric'|'categorical'}")
    ap.add_argument("--model", choices=["lr","svm"], default="lr", help="Classifier to train")
    ap.add_argument("--neg_per_pos", type=int, default=3, help="Negatives per positive when building pairs")
    ap.add_argument("--text_m", type=int, default=512, help="CBF length")
    ap.add_argument("--text_k", type=int, default=3,   help="CBF hash count")
    ap.add_argument("--text_q", type=int, default=2,   help="q-gram length")
    ap.add_argument("--salt", default="mltext", help="Salt for CBF hashing")
    ap.add_argument("--out", help="Optional CSV file for test-set predictions/features")
    ap.add_argument("--metrics", help="Optional JSON file with metrics")
    args = ap.parse_args()

    A = pd.read_csv(args.alice); B = pd.read_csv(args.bob)

    if args.schema and os.path.exists(args.schema):
        schema = json.load(open(args.schema,"r",encoding="utf-8"))
    else:
        schema = auto_schema(A, B, args.idA, args.idB)
        print("Inferred schema:\n", json.dumps(schema, indent=2))

    pairs = build_pairs(A, B, idA=args.idA, idB=args.idB, neg_per_pos=args.neg_per_pos, seed=42)
    F = assemble_features(A, B, pairs, schema,
                          idA=args.idA, idB=args.idB,
                          text_m=args.text_m, text_k=args.text_k, text_q=args.text_q, salt=args.salt)

    ml, base = train_and_eval(F, model=args.model)
    print("\n=== Baseline (threshold-like) ===")
    print(f"Precision: {base['precision']:.4f}  Recall: {base['recall']:.4f}  F1: {base['f1']:.4f}")
    print(f"\n=== ML ({args.model.upper()}) ===")
    print(f"Precision: {ml['precision']:.4f}  Recall: {ml['recall']:.4f}  F1: {ml['f1']:.4f}")
    print(f"\nΔF1 (ML − Baseline): {ml['f1'] - base['f1']:+.4f}")

    # save optional artifacts using the same split for reproducibility
    if args.out or args.metrics:
        feat_cols = [c for c in F.columns if c not in ("label","idA","idB")]
        X = F[feat_cols].values; y = F["label"].values.astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=11, stratify=y)

        if args.model == "lr":
            est = LogisticRegression(max_iter=1000)
        else:
            est = make_pipeline(StandardScaler(with_mean=False),
                                SVC(kernel="rbf", class_weight="balanced", C=2.0, gamma="scale"))
        est.fit(Xtr, ytr)
        yhat_ml = est.predict(Xte)
        te_df = pd.DataFrame(Xte, columns=feat_cols)
        te_df["y_true"] = yte
        te_df["y_pred_ml"] = yhat_ml
        te_df["y_pred_baseline"] = baseline_predict(te_df)

        if args.out:
            out_dir = os.path.dirname(args.out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            te_df.to_csv(args.out, index=False)
            print(f"Saved predictions/features → {args.out}")

        if args.metrics:
            metrics_dir = os.path.dirname(args.metrics)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            with open(args.metrics, "w") as f:
                json.dump({"ml": ml, "baseline": base, "delta_f1": ml["f1"] - base["f1"]}, f, indent=2)
            print(f"Saved metrics JSON → {args.metrics}")

if __name__ == "__main__":
    main()
