# logistic_regression_linkage.py
# Simple, fast-ish baseline: TF-IDF(char 3-4) -> SVD -> pairwise features -> LogisticRegression
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import joblib


# ------------------------- utils -------------------------
def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return " ".join(s.lower().split())


@dataclass
class Config:
    char_min: int = 3
    char_max: int = 4           # keep smaller for speed
    n_features: int = 2 ** 17   # 131,072 hashed features (fast)
    svd_dims: int = 128         # set higher (256/384) later for more accuracy
    random_state: int = 42
    val_size: float = 0.2
    C: float = 1.0              # LogisticRegression regularization
    max_iter: int = 200


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
        self.svd = TruncatedSVD(
            n_components=cfg.svd_dims,
            n_iter=3,
            random_state=cfg.random_state
        )
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


class PairwiseMLLinker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embed = TextEmbedder(cfg)  # share one embedder for both sides (faster)
        self.clf = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, solver="lbfgs", n_jobs=1)
        self.tau_ = 0.5

    def _prepare_Xy(self, df: pd.DataFrame, text1: str, text2: str, label: str, fit_embedder: bool) -> Tuple[np.ndarray, np.ndarray]:
        t1 = df[text1].astype(str).values
        t2 = df[text2].astype(str).values
        if fit_embedder:
            self.embed.fit(np.concatenate([t1, t2], axis=0))
        len1 = np.array([len(norm_text(x)) for x in t1], dtype=np.float32)
        len2 = np.array([len(norm_text(x)) for x in t2], dtype=np.float32)
        u = self.embed.transform(t1)
        v = self.embed.transform(t2)
        X = pair_features(u, v, len1, len2)
        y = df[label].astype(int).values
        return X, y

    def fit(self, train_df: pd.DataFrame, text1: str, text2: str, label: str) -> Dict:
        cfg = self.cfg
        tr, va = train_test_split(train_df, test_size=cfg.val_size, stratify=train_df[label], random_state=cfg.random_state)
        X_tr, y_tr = self._prepare_Xy(tr, text1, text2, label, fit_embedder=True)
        X_va, y_va = self._prepare_Xy(va, text1, text2, label, fit_embedder=False)

        self.clf.fit(X_tr, y_tr)

        # tune threshold for F1
        scores = self.clf.predict_proba(X_va)[:, 1]
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.1, 0.9, 81):
            pred = (scores >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_va, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.tau_ = float(best_t)
        return {"tau": self.tau_, "val_f1": float(best_f1)}

    def evaluate(self, test_df: pd.DataFrame, text1: str, text2: str, label: str) -> Dict:
        X_te, y_te = self._prepare_Xy(test_df, text1, text2, label, fit_embedder=False)
        scores = self.clf.predict_proba(X_te)[:, 1]
        pr_auc = float(average_precision_score(y_te, scores))
        pred = (scores >= self.tau_).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
        return {"pr_auc": pr_auc, "precision": float(p), "recall": float(r), "f1": float(f1)}

    def save(self, path: str):
        # for compatibility with earlier evaluator: save embed twice under embed1/embed2
        joblib.dump({"cfg": asdict(self.cfg), "embed1": self.embed, "embed2": self.embed, "clf": self.clf, "tau": self.tau_}, path)


# ------------------------- CLI -------------------------
if __name__ == "__main__":
    
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    DEFAULT_TRAIN = ROOT / "csv_files" / "target_train.csv"
    DEFAULT_TEST = ROOT / "csv_files" / "target_test.csv"
    DEFAULT_MODEL = HERE / "ml_linkage_baseline.joblib"

    import argparse
    ap = argparse.ArgumentParser(description="Simple ML record linkage baseline")
    ap.add_argument("--train_csv", type=str, default=str(DEFAULT_TRAIN))
    ap.add_argument("--test_csv", type=str, default=str(DEFAULT_TEST))
    ap.add_argument("--text1", type=str, default="text1")
    ap.add_argument("--text2", type=str, default="text2")
    ap.add_argument("--label", type=str, default="label")
    ap.add_argument("--svd_dims", type=int, default=Config.svd_dims)
    ap.add_argument("--C", type=float, default=Config.C)
    ap.add_argument("--model_out", type=str, default=str(DEFAULT_MODEL))
    args = ap.parse_args()

    # config
    cfg = Config(svd_dims=args.svd_dims, C=args.C)
    model = PairwiseMLLinker(cfg)

    # data
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    info = model.fit(train_df, args.text1, args.text2, args.label)
    metrics = model.evaluate(test_df, args.text1, args.text2, args.label)
    model.save(args.model_out)

    out = {"config": asdict(cfg), "train": info, "test": metrics, "model_path": args.model_out}
    print(json.dumps(out, indent=2))
