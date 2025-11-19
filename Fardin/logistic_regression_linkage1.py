# logistic_regression_linkage_dp.py
# Same pipeline as baseline, but with (ε, δ)-DP via Gaussian noise on L2-clipped embeddings.
# - Supports sweeping multiple epsilons (comma-separated) and prints metrics per ε.
# - Default paths match  repo: csv_files/{target_train.csv,target_test.csv}

from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,           # AUCPR
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib

# --------------- utils  ---------------
def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return " ".join(s.lower().split())

# Inputs tables with "text1, text2, label" columns (1= match, 0=no-match))

def pair_features(u: np.ndarray, v: np.ndarray, len1: np.ndarray, len2: np.ndarray) -> np.ndarray:
    abs_diff = np.abs(u - v)
    hadamard = u * v
    u_norm = np.linalg.norm(u, axis=1) + 1e-12
    v_norm = np.linalg.norm(v, axis=1) + 1e-12
    cos = ((u * v).sum(axis=1) / (u_norm * v_norm)).reshape(-1, 1)
    len1 = len1.reshape(-1, 1)
    len2 = len2.reshape(-1, 1)
    len_diff = np.abs(len1 - len2)
    len_ratio = (np.minimum(len1, len2) / np.maximum(len1, len2)).astype(np.float32)
    return np.hstack([abs_diff, hadamard, cos, len1, len2, len_diff, len_ratio])

# --------------- config ---------------
@dataclass
class Config:
    # feature pipeline (kept light for speed)
    char_min: int = 3
    char_max: int = 4
    n_features: int = 2 ** 17
    svd_dims: int = 128
    random_state: int = 42
    val_size: float = 0.2
    C: float = 1.0
    max_iter: int = 300

    # Differential Privacy controls
    epsilon: float = float("inf")   # inf => no DP (baseline)
    delta: float = 1e-5
    clip_norm: float = 1.0          # L2 clip before adding noise

# --------------- embedder  ---------------
class TextEmbedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hv = HashingVectorizer(
            analyzer="char",
            ngram_range=(cfg.char_min, cfg.char_max),
            n_features=cfg.n_features,
            alternate_sign=False,
            norm=None,
            lowercase=False,
        )
        self.tfidf = TfidfTransformer(norm="l2", use_idf=True)
        self.svd = TruncatedSVD(n_components=cfg.svd_dims, n_iter=3, random_state=cfg.random_state)
        self.fitted_ = False

    def fit(self, texts: np.ndarray):
        texts = [norm_text(t) for t in texts]
        X = self.hv.transform(texts)
        X = self.tfidf.fit_transform(X)
        X = self.svd.fit_transform(X)
        self.fitted_ = True
        return self

    def transform(self, texts: np.ndarray) -> np.ndarray:
        assert self.fitted_, "TextEmbedder not fitted"
        texts = [norm_text(t) for t in texts]
        X = self.hv.transform(texts)
        X = self.tfidf.transform(X)
        X = self.svd.transform(X)
        return X.astype(np.float32)

# --------------- Differential Privacy helpers ---------------
def _gaussian_sigma(eps: float, delta: float, clip_norm: float) -> float:
    """Sigma for (ε, δ)-DP Gaussian mechanism with L2-sensitivity=clip_norm."""
    if not np.isfinite(eps) or eps <= 0:
        return 0.0  # treat as 'no DP' if eps is inf or invalid
    return clip_norm * math.sqrt(2.0 * math.log(1.25 / delta)) / eps

def _clip_l2(X: np.ndarray, C: float) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    scale = np.minimum(1.0, C / norms)
    return X * scale

def _privatise_embeddings(X: np.ndarray, clip_norm: float, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    Xc = _clip_l2(X, clip_norm)
    if sigma > 0.0:
        noise = rng.normal(0.0, sigma, size=Xc.shape).astype(np.float32)
        Xc = Xc + noise
    return Xc

# --------------- model ---------------
class PairwiseMLLinkerDP:
    """PairwiseMLLinkage, but injects DP noise on embeddings."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embed = TextEmbedder(cfg)   # share one embedder for both sides
        self.clf = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, solver="lbfgs", n_jobs=1)
        self.tau_ = 0.5
        self._rng = np.random.RandomState(cfg.random_state)

    def _prepare_Xy(self, df: pd.DataFrame, text1: str, text2: str, label: str, fit_embedder: bool) -> Tuple[np.ndarray, np.ndarray]:
        t1 = df[text1].astype(str).values
        t2 = df[text2].astype(str).values
        if fit_embedder:
            self.embed.fit(np.concatenate([t1, t2], axis=0))

        # Base embeddings
        u_raw = self.embed.transform(t1)
        v_raw = self.embed.transform(t2)

        # DP step: clip + Gaussian noise
        sigma = _gaussian_sigma(self.cfg.epsilon, self.cfg.delta, self.cfg.clip_norm)
        u = _privatise_embeddings(u_raw, self.cfg.clip_norm, sigma, self._rng)
        v = _privatise_embeddings(v_raw, self.cfg.clip_norm, sigma, self._rng)

        # Pairwise features
        len1 = np.array([len(norm_text(x)) for x in t1], dtype=np.float32)
        len2 = np.array([len(norm_text(x)) for x in t2], dtype=np.float32)
        X = pair_features(u, v, len1, len2)
        y = df[label].astype(int).values
        return X, y

    def fit(self, train_df: pd.DataFrame, text1: str, text2: str, label: str) -> Dict:
        cfg = self.cfg
        tr, va = train_test_split(train_df, test_size=cfg.val_size, stratify=train_df[label], random_state=cfg.random_state)
        X_tr, y_tr = self._prepare_Xy(tr, text1, text2, label, fit_embedder=True)
        X_va, y_va = self._prepare_Xy(va, text1, text2, label, fit_embedder=False)

        self.clf.fit(X_tr, y_tr)

        # tune threshold on validation for F1
        scores = self.clf.predict_proba(X_va)[:, 1]
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.1, 0.9, 81):
            pred = (scores >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_va, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.tau_ = float(best_t)

        return {
            "tau": self.tau_,
            "val_f1": float(best_f1),
            "epsilon": float(cfg.epsilon),
            "delta": float(cfg.delta),
            "clip_norm": float(cfg.clip_norm),
            "sigma": float(_gaussian_sigma(cfg.epsilon, cfg.delta, cfg.clip_norm)),
        }

    def evaluate(self, test_df: pd.DataFrame, text1: str, text2: str, label: str) -> Dict:
        X_te, y_te = self._prepare_Xy(test_df, text1, text2, label, fit_embedder=False)
        scores = self.clf.predict_proba(X_te)[:, 1]
        pred = (scores >= self.tau_).astype(int)
        pr_auc = float(average_precision_score(y_te, scores))  # AUCPR
        p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
        acc = float(accuracy_score(y_te, pred))
        roc = float(roc_auc_score(y_te, scores))
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        return {
            "pr_auc": pr_auc,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "accuracy": acc,
            "roc_auc": roc,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "n": int(len(y_te)),
        }

# --------------- CLI ---------------
if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    DEFAULT_TRAIN = ROOT / "csv_files" / "target_train.csv"
    DEFAULT_TEST  = ROOT / "csv_files" / "target_test.csv"

    import argparse
    ap = argparse.ArgumentParser(description="LogReg linkage with DP on embeddings (prints AUCPR per ε)")
    ap.add_argument("--train_csv", type=str, default=str(DEFAULT_TRAIN))
    ap.add_argument("--test_csv",  type=str, default=str(DEFAULT_TEST))
    ap.add_argument("--text1", type=str, default="text1")
    ap.add_argument("--text2", type=str, default="text2")
    ap.add_argument("--label", type=str, default="label")
    ap.add_argument("--svd_dims", type=int, default=Config.svd_dims)
    ap.add_argument("--C", type=float, default=Config.C)
    ap.add_argument("--clip_norm", type=float, default=Config.clip_norm)
    ap.add_argument("--delta", type=float, default=Config.delta)
    ap.add_argument("--epsilons", type=str, default="inf,5,2,1,0.5,0.25",
                    help="Comma-separated ε values. Use 'inf' for no-DP baseline.")
    args = ap.parse_args()

    # data once; re-use across ε
    train_df = pd.read_csv(args.train_csv)
    test_df  = pd.read_csv(args.test_csv)

    # run a sweep
    results: List[Dict] = []
    for eps_str in [s.strip() for s in args.epsilons.split(",") if s.strip()]:
        epsilon = float("inf") if eps_str.lower() in ("inf", "none", "baseline") else float(eps_str)
        cfg = Config(
            svd_dims=args.svd_dims, C=args.C,
            epsilon=epsilon, delta=args.delta, clip_norm=args.clip_norm
        )
        model = PairwiseMLLinkerDP(cfg)
        info = model.fit(train_df, args.text1, args.text2, args.label)
        metrics = model.evaluate(test_df, args.text1, args.text2, args.label)
        row = {"epsilon": float(epsilon if np.isfinite(epsilon) else np.inf), **info, "test": metrics}
        results.append(row)

        # headline per ε
        eps_label = "inf" if not np.isfinite(epsilon) else f"{epsilon:g}"
        print(f"[ε={eps_label}] AUCPR={metrics['pr_auc']:.6f}  F1={metrics['f1']:.4f}  "
              f"prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}  (sigma={info['sigma']:.4f})")

    # final JSON dump (compact)
    print(json.dumps({"runs": results}, indent=2))
