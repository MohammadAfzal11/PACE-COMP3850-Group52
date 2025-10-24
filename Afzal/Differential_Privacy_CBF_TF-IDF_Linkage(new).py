import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import hashlib
import json
import time
import warnings
warnings.filterwarnings('ignore')


# =========================================================
# Differential-Privacy Counting Bloom Filter (DP-CBF)
# =========================================================
class WorkingDifferentialPrivacyCBF:
    """DP-CBF for text linkage with evaluation"""

    def __init__(self, bf_len=1000, num_hash_func=10, q=2, epsilon=1.0):
        self.bf_len = bf_len
        self.num_hash_func = num_hash_func
        self.q = q
        self.epsilon = epsilon
        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5

    def _normalize(self, s):
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return ""
        return str(s).lower().strip()

    def get_qgrams(self, text):
        text = self._normalize(text)
        if len(text) < self.q:
            return [text.ljust(self.q, ' ')]
        return [text[i:i+self.q] for i in range(len(text) - self.q + 1)]

    def encode_record_clean(self, record, fields=('raw_text',)):
        cbf = np.zeros(self.bf_len, dtype=int)
        for field in fields:
            if field in record and record[field] is not None:
                qgrams = self.get_qgrams(record[field])
                for qg in qgrams:
                    int1 = int(self.h1(qg.encode('utf-8')).hexdigest(), 16)
                    int2 = int(self.h2(qg.encode('utf-8')).hexdigest(), 16)
                    for i in range(self.num_hash_func):
                        gi = (int1 + i * int2) % self.bf_len
                        cbf[gi] += 1
        return cbf

    def add_calibrated_noise(self, cbf):
        # Laplace(0, Δf/ε)
        sensitivity = 0.1
        b = sensitivity / self.epsilon if self.epsilon > 0 else 0.01
        noise = np.random.laplace(0, b, size=cbf.shape)
        noisy = cbf + noise
        return np.maximum(0, noisy).astype(float)

    def encode_record_private(self, record, fields=('raw_text',)):
        return self.add_calibrated_noise(self.encode_record_clean(record, fields))

    @staticmethod
    def dice_similarity(cbf1, cbf2):
        sum1, sum2 = np.sum(cbf1), np.sum(cbf2)
        if sum1 + sum2 == 0:
            return 0.0
        common = np.sum(np.minimum(cbf1, cbf2))
        return (2.0 * common) / (sum1 + sum2)


# =========================================================
# Normalization and TF-IDF (character n-grams)
# =========================================================
def normalize_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).lower().strip()
    s = " ".join(s.split())  # compact whitespace
    return s

def fit_tfidf_char(train_texts, ngram_range=(2,5), min_df=5, max_features=200_000):
    # Control dimensionality to keep memory reasonable
    vec = TfidfVectorizer(
        analyzer='char',
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.95,
        max_features=max_features,
        lowercase=False,
        norm='l2',
        dtype=np.float32
    )
    X = vec.fit_transform(train_texts)
    return vec, X

def tfidf_cosine_pairs_sparse(vectorizer, pairs, epsilon=1.0, dp=True, sensitivity=0.05, batch_size=5000):
    """
    Compute cosine similarities fully sparse.
    Apply DP by adding Laplace noise to each scalar cosine score (output perturbation).
    """
    n = len(pairs)
    sims = np.zeros(n, dtype=np.float32)
    b = sensitivity / epsilon if epsilon > 0 else 0.01

    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        texts1 = [pairs[k][0]['raw_text'] for k in range(i, j)]
        texts2 = [pairs[k][1]['raw_text'] for k in range(i, j)]

        X1 = vectorizer.transform(texts1)   # csr_matrix
        X2 = vectorizer.transform(texts2)   # csr_matrix

        # L2 normalize in sparse; no densification
        X1n = normalize(X1, norm='l2', copy=False)
        X2n = normalize(X2, norm='l2', copy=False)

        # Rowwise cosine via elementwise multiply then sum per row
        block = X1n.multiply(X2n).sum(axis=1).A1  # A1 => flat ndarray

        if dp:
            block = block + np.random.laplace(0.0, b, size=block.shape)

        sims[i:j] = np.clip(block, 0.0, 1.0)

    return sims


# =========================================================
# Pair builders for text-pair datasets (text1, text2, label)
# =========================================================
def load_textpair_csv(path, nrows=None):
    return pd.read_csv(path, nrows=nrows)

def create_pairs_labels_from_textpair(df):
    pairs, labels = [], []
    for _, r in df.iterrows():
        rec1 = {"raw_text": normalize_text(r.get("text1", ""))}
        rec2 = {"raw_text": normalize_text(r.get("text2", ""))}
        pairs.append((rec1, rec2))
        labels.append(int(r["label"]))
    return pairs, np.array(labels, dtype=np.int64)


# =========================================================
# DP-CBF similarities, fusion, threshold tuning, evaluation
# =========================================================
def dp_cbf_similarities(pairs, epsilon=1.0, bf_len=1000, num_hash_func=10, q=2):
    dp = WorkingDifferentialPrivacyCBF(bf_len=bf_len, num_hash_func=num_hash_func, q=q, epsilon=epsilon)
    sims_priv = np.zeros(len(pairs), dtype=np.float32)
    sims_clean = np.zeros(len(pairs), dtype=np.float32)
    for i, (r1, r2) in enumerate(pairs):
        c1p = dp.encode_record_private(r1)
        c2p = dp.encode_record_private(r2)
        sims_priv[i] = dp.dice_similarity(c1p, c2p)

        c1c = dp.encode_record_clean(r1)
        c2c = dp.encode_record_clean(r2)
        sims_clean[i] = dp.dice_similarity(c1c, c2c)
    return sims_priv, sims_clean

def tune_fusion_threshold(labels, dice, cos, alpha_grid=None, thr_grid=None, metric="f1"):
    if alpha_grid is None:
        alpha_grid = np.linspace(0.2, 0.8, 13)  # 0.2..0.8 step 0.05
    if thr_grid is None:
        thr_grid = np.linspace(0.05, 0.95, 19)

    best = {"alpha":0.5, "thr":0.5, "f1":0.0, "acc":0.0, "bacc":0.0}
    for a in alpha_grid:
        fused = a * dice + (1 - a) * cos
        for t in thr_grid:
            pred = (fused > t).astype(int)
            f1 = f1_score(labels, pred)
            acc = accuracy_score(labels, pred)
            bacc = balanced_accuracy_score(labels, pred)
            score = f1 if metric == "f1" else bacc
            if score > best[metric]:
                best.update({"alpha":float(a), "thr":float(t), "f1":float(f1), "acc":float(acc), "bacc":float(bacc)})
    return best

def evaluate_predictions(labels, scores, thr, title=None):
    pred = (scores > thr).astype(int)
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred)
    if title:
        print(f"\n{title}")
    print(classification_report(labels, pred))
    print("Confusion matrix:\n", confusion_matrix(labels, pred))
    return acc, f1


# =========================================================
# Main experiment
# =========================================================
def main():
    np.random.seed(42)

    # Inputs (adjust paths as needed)
    train_path = r"C:\Users\afzal\Documents\GitHub\PACE-COMP3850-Group52\csv_files\target_train.csv"
    test_path  = r"C:\Users\afzal\Documents\GitHub\PACE-COMP3850-Group52\csv_files\target_test.csv"

    print("Loading data...")
    train_df = load_textpair_csv(train_path)
    test_df  = load_textpair_csv(test_path)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Build (text1, text2, label) pairs
    train_pairs, train_labels = create_pairs_labels_from_textpair(train_df)
    test_pairs,  test_labels  = create_pairs_labels_from_textpair(test_df)

    # Prepare TF-IDF on training texts only (no leakage)
    print("\nFitting TF-IDF vectorizer on training corpus...")
    train_texts = [p[0]['raw_text'] for p in train_pairs] + [p[1]['raw_text'] for p in train_pairs]
    vec, _ = fit_tfidf_char(train_texts, ngram_range=(2,5), min_df=5, max_features=200_000)
    print(f"TF-IDF vocabulary size: {len(vec.vocabulary_)}")

    # Epsilon sweep
    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    start = time.time()

    for eps in epsilons:
        print(f"\n=== Epsilon={eps} ===")

        # 1) DP-CBF similarities
        print("Computing DP-CBF similarities (train)...")
        train_dice_dp, train_dice_clean = dp_cbf_similarities(train_pairs, epsilon=eps, bf_len=1000, num_hash_func=10, q=2)

        print("Computing DP-CBF similarities (test)...")
        test_dice_dp,  test_dice_clean  = dp_cbf_similarities(test_pairs,  epsilon=eps, bf_len=1000, num_hash_func=10, q=2)

        # 2) TF-IDF cosine similarities (DP on scalar scores, memory-safe)
        print("Computing TF-IDF cosine (train)...")
        train_cos_dp  = tfidf_cosine_pairs_sparse(vec, train_pairs, epsilon=eps, dp=True,  sensitivity=0.05, batch_size=5000)
        train_cos_cln = tfidf_cosine_pairs_sparse(vec, train_pairs, epsilon=eps, dp=False, sensitivity=0.05, batch_size=5000)

        print("Computing TF-IDF cosine (test)...")
        test_cos_dp   = tfidf_cosine_pairs_sparse(vec, test_pairs,  epsilon=eps, dp=True,  sensitivity=0.05, batch_size=5000)
        test_cos_cln  = tfidf_cosine_pairs_sparse(vec, test_pairs,  epsilon=eps, dp=False, sensitivity=0.05, batch_size=5000)

        # 3) Tune fusion and threshold on train (DP branches)
        print("Tuning fusion and threshold on training set (DP branches)...")
        best = tune_fusion_threshold(
            train_labels, train_dice_dp, train_cos_dp,
            alpha_grid=np.linspace(0.2, 0.8, 13),
            thr_grid=np.linspace(0.05, 0.95, 19),
            metric="f1"  # use "bacc" to optimize balanced accuracy instead
        )
        print(f"Best fusion (train, DP): alpha={best['alpha']:.2f}, thr={best['thr']:.2f}, "
              f"F1={best['f1']:.4f}, Acc={best['acc']:.4f}, BAcc={best['bacc']:.4f}")

        # 4) Evaluate on test with tuned alpha and thr (DP fused and branches)
        fused_test_dp = best['alpha']*test_dice_dp + (1-best['alpha'])*test_cos_dp
        acc_fused_dp, f1_fused_dp = evaluate_predictions(test_labels, fused_test_dp, best['thr'], title="DP Fused (Test)")

        acc_dice_dp, f1_dice_dp = evaluate_predictions(test_labels, test_dice_dp, best['thr'], title="DP Dice (Test)")
        acc_cos_dp,  f1_cos_dp  = evaluate_predictions(test_labels, test_cos_dp,  best['thr'], title="DP TF-IDF (Test)")

        # 5) Clean references (utility gap)
        fused_test_cln = best['alpha']*test_dice_clean + (1-best['alpha'])*test_cos_cln
        acc_fused_cln, f1_fused_cln = evaluate_predictions(test_labels, fused_test_cln, best['thr'], title="Clean Fused (Test)")
        acc_dice_cln,  f1_dice_cln  = evaluate_predictions(test_labels, test_dice_clean, best['thr'], title="Clean Dice (Test)")
        acc_cos_cln,   f1_cos_cln   = evaluate_predictions(test_labels, test_cos_cln,  best['thr'], title="Clean TF-IDF (Test)")

        result = {
            "epsilon": eps,
            "alpha": best['alpha'],
            "thr": best['thr'],
            "dp_fused_acc": float(acc_fused_dp),
            "dp_fused_f1":  float(f1_fused_dp),
            "dp_dice_acc":  float(acc_dice_dp),
            "dp_dice_f1":   float(f1_dice_dp),
            "dp_cos_acc":   float(acc_cos_dp),
            "dp_cos_f1":    float(f1_cos_dp),
            "cln_fused_acc": float(acc_fused_cln),
            "cln_fused_f1":  float(f1_fused_cln),
            "cln_dice_acc":  float(acc_dice_cln),
            "cln_dice_f1":   float(f1_dice_cln),
            "cln_cos_acc":   float(acc_cos_cln),
            "cln_cos_f1":    float(f1_cos_cln)
        }
        results.append(result)

        print(f"\nSUMMARY @ ε={eps}:")
        print(f"DP Fused    - Acc={acc_fused_dp:.4f}, F1={f1_fused_dp:.4f}")
        print(f"DP Dice     - Acc={acc_dice_dp:.4f}, F1={f1_dice_dp:.4f}")
        print(f"DP TF-IDF   - Acc={acc_cos_dp:.4f},  F1={f1_cos_dp:.4f}")
        print(f"Clean Fused - Acc={acc_fused_cln:.4f}, F1={f1_fused_cln:.4f}")

    runtime = time.time() - start

    # =======================
    # Print tabular summary
    # =======================
    print("\n" + "="*90)
    print("PRIVACY-UTILITY SUMMARY (DP fused vs branches, plus clean refs)")
    print("="*90)
    header = f"{'eps':<6}{'alpha':>7}{'thr':>7} | {'dp_fused':>9}{'dp_dice':>9}{'dp_tfidf':>10} | {'cln_fused':>9}"
    print(header)
    print("-"*90)
    for r in results:
        print(f"{r['epsilon']:<6}{r['alpha']:>7.2f}{r['thr']:>7.2f} | "
              f"{r['dp_fused_f1']:>9.4f}{r['dp_dice_f1']:>9.4f}{r['dp_cos_f1']:>10.4f} | "
              f"{r['cln_fused_f1']:>9.4f}")

    # =======================
    # Plot (DP fused vs eps)
    # =======================
    eps_list = [r["epsilon"] for r in results]
    dp_fused = [r["dp_fused_f1"] for r in results]
    cln_fused = [r["cln_fused_f1"] for r in results]

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(eps_list, dp_fused, 'g-o', label='DP Fused F1')
    plt.plot(eps_list, cln_fused, 'k--s', label='Clean Fused F1')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('F1 Score')
    plt.title('Privacy-Utility (Fused) across ε')
    plt.grid(True, alpha=0.3); plt.legend()

    dp_dice_f1  = [r["dp_dice_f1"] for r in results]
    dp_cos_f1   = [r["dp_cos_f1"] for r in results]
    plt.subplot(1,2,2)
    plt.plot(eps_list, dp_dice_f1, 'b-o', label='DP Dice (CBF)')
    plt.plot(eps_list, dp_cos_f1,  'm--s', label='DP TF-IDF Cosine')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('F1 Score')
    plt.title('Branch Performance across ε')
    plt.grid(True, alpha=0.3); plt.legend()

    plt.tight_layout()
    plt.savefig('dp_cbf_tfidf_fusion_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved figure: dp_cbf_tfidf_fusion_summary.png")

    # Save results
    with open('dp_cbf_tfidf_fusion_results.json', 'w') as f:
        json.dump({"results": results, "runtime_sec": runtime}, f, indent=2)
    print("Saved results: dp_cbf_tfidf_fusion_results.json")


if __name__ == "__main__":
    main()
