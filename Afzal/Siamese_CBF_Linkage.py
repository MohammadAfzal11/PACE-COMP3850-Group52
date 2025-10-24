import os, re, gc, hashlib
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# NEW: imports for figures
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

# ----------------------------
# Normalization & helpers
# ----------------------------
def norm_string(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return " ".join(str(x).lower().strip().split())

def join_fields(values: List[str]) -> str:
    return " ".join([norm_string(v) for v in values if norm_string(v) != ""])

# ----------------------------
# Multi-encoding Counting Bloom Filter
# ----------------------------
class CountingBloomFilter:
    def __init__(self, bf_len=1000, num_hash_func=10, q=2, max_count_cap=5):
        self.bf_len = bf_len
        self.num_hash_func = num_hash_func
        self.q = q
        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5
        self.max_count_cap = max_count_cap

    def qgrams(self, text: str):
        text = norm_string(text)
        if len(text) < self.q:
            return [text.ljust(self.q, ' ')]
        return [text[i:i+self.q] for i in range(len(text)-self.q+1)]

    def _hash_positions(self, token: str, salt: Optional[bytes]=None):
        t = token.encode('utf-8')
        if salt:
            a = self.h1(salt + t).hexdigest()
            b = self.h2(t + salt).hexdigest()
        else:
            a = self.h1(t).hexdigest()
            b = self.h2(t).hexdigest()
        int1 = int(a, 16); int2 = int(b, 16)
        for i in range(self.num_hash_func):
            yield (int1 + i * int2) % self.bf_len

    def encode_text(self, text: str, salt: Optional[str]=None) -> np.ndarray:
        cbf = np.zeros(self.bf_len, dtype=np.int32)
        sbytes = salt.encode('utf-8') if salt else None
        for qg in self.qgrams(text):
            for gi in self._hash_positions(qg, sbytes):
                cbf[gi] += 1
        return cbf

    def post(self, v: np.ndarray) -> np.ndarray:
        v = np.clip(v, 0, self.max_count_cap).astype(np.float32)
        v = np.log1p(v)
        n = np.linalg.norm(v) + 1e-8
        return (v / n).astype(np.float32)

    def encode_multi_channels(self,
                              text: str,
                              salt2: str = "salt-v1",
                              add_binary: bool = False,
                              extra_q: Optional[int] = None) -> np.ndarray:
        # base q
        c1 = self.post(self.encode_text(text, salt=None))
        c2 = self.post(self.encode_text(text, salt=salt2))
        chans = [c1, c2]
        # optional: binary channel from base q
        if add_binary:
            b = (c1 > 0).astype(np.float32)
            b = b / (np.linalg.norm(b) + 1e-8)
            chans.append(b)
        # optional: extra q channel (unsalted+salted) to capture longer patterns
        if extra_q is not None and extra_q != self.q:
            q_old = self.q
            self.q = extra_q
            c3 = self.post(self.encode_text(text, salt=None))
            c4 = self.post(self.encode_text(text, salt=salt2))
            chans.extend([c3, c4])
            self.q = q_old
        return np.concatenate(chans, axis=0).astype(np.float32)

# ----------------------------
# Pair parsing for whole records
# ----------------------------
def infer_side_columns(df: pd.DataFrame,
                       label_col: str = "label") -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c != label_col]
    if "text1" in df.columns and "text2" in df.columns:
        return ["text1"], ["text2"]
    left = [c for c in cols if re.match(r'^(left_|l_|src_|a_).+', c, flags=re.I)]
    right = [c for c in cols if re.match(r'^(right_|r_|dst_|b_).+', c, flags=re.I)]
    if left and right:
        return left, right
    left = [c for c in cols if re.search(r'(_1|1)$', c)]
    right = [c for c in cols if re.search(r'(_2|2)$', c)]
    if left and right:
        return left, right
    raise ValueError("Unable to infer left/right columns; provide text1/text2 or left_/right_ prefixes.")

def build_pair_texts(df: pd.DataFrame,
                     label_col: str = "label") -> Tuple[List[str], List[str], np.ndarray]:
    L, R = infer_side_columns(df, label_col=label_col)
    lt, rt, y = [], [], []
    for _, r in df.iterrows():
        lt.append(join_fields([r[c] for c in L if c in df.columns]))
        rt.append(join_fields([r[c] for c in R if c in df.columns]))
        y.append(int(r[label_col]))
    return lt, rt, np.array(y, dtype=np.int64)

def encode_pairs_cbf(left_texts: List[str], right_texts: List[str],
                     encoder: CountingBloomFilter,
                     salt2: str = "salt-v1",
                     add_binary: bool = False,
                     extra_q: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    X1 = np.stack([encoder.encode_multi_channels(t, salt2=salt2,
                                                add_binary=add_binary,
                                                extra_q=extra_q) for t in left_texts]).astype(np.float32)
    X2 = np.stack([encoder.encode_multi_channels(t, salt2=salt2,
                                                add_binary=add_binary,
                                                extra_q=extra_q) for t in right_texts]).astype(np.float32)
    gc.collect()
    return X1, X2

# NEW: single-channel packing to create a "before" baseline at same input shape
def encode_pairs_cbf_single_channel(left_texts: List[str], right_texts: List[str],
                                    encoder: CountingBloomFilter,
                                    total_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    def pack_single(text: str):
        u = encoder.post(encoder.encode_text(text, salt=None))
        pads = [np.zeros_like(u) for _ in range(total_channels - 1)]
        return np.concatenate([u] + pads, axis=0).astype(np.float32)
    X1 = np.stack([pack_single(t) for t in left_texts]).astype(np.float32)
    X2 = np.stack([pack_single(t) for t in right_texts]).astype(np.float32)
    return X1, X2

# ----------------------------
# Siamese model
# ----------------------------
class SiameseCBF:
    def __init__(self, input_dim: int, embedding_dim=128, l2=1e-4, dropout=0.3,
                 use_cosine_head=False):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.l2 = l2
        self.dropout = dropout
        self.use_cosine_head = use_cosine_head
        self.model = None

    def tower(self):
        return models.Sequential([
            layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(self.l2)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(self.l2)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout/1.5),
            layers.Dense(self.embedding_dim, activation="relu", kernel_regularizer=regularizers.l2(self.l2)),
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2norm")
        ])

    def build(self):
        x1 = layers.Input(shape=(self.input_dim,), name="cbf_1")
        x2 = layers.Input(shape=(self.input_dim,), name="cbf_2")
        enc = self.tower()
        e1, e2 = enc(x1), enc(x2)

        if self.use_cosine_head:
            cos = layers.Dot(axes=1, normalize=True)([e1, e2])
            h = layers.Concatenate()([cos,])
            h = layers.Dense(32, activation="relu")(h)
            out = layers.Dense(1, activation="sigmoid")(h)
        else:
            diff = layers.Lambda(lambda xs: tf.abs(xs[0]-xs[1]))([e1, e2])
            mult = layers.Lambda(lambda xs: xs[0]*xs[1])([e1, e2])
            h = layers.Concatenate()([diff, mult])
            h = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(self.l2))(h)
            h = layers.Dropout(self.dropout)(h)
            h = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(self.l2))(h)
            out = layers.Dense(1, activation="sigmoid")(h)

        self.model = models.Model(inputs=[x1, x2], outputs=out)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                           loss="binary_crossentropy",
                           metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                                    tf.keras.metrics.Recall(name="recall")])
        return self.model

# ----------------------------
# Threshold tuning
# ----------------------------
def tune_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray, grid=None):
    grid = grid or np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, 0.0
    for t in grid:
        yp = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, yp)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr), float(best_f1)

# ----------------------------
# NEW: Figure helpers (Boundary, ROC+Youden, Before/After CMs)
# ----------------------------
def save_boundary_density(scores, labels, thr, title, fname):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    plt.figure(figsize=(6,4))
    plt.hist(neg, bins=40, alpha=0.5, label='Non-match', density=True)
    plt.hist(pos, bins=40, alpha=0.5, label='Match', density=True)
    plt.axvline(thr, color='k', linestyle='--', label=f'Threshold={thr:.2f}')
    plt.xlabel('Siamese match score'); plt.ylabel('Density'); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close()

def save_roc_with_youden(scores, labels, title, fname):
    fpr, tpr, thr = roc_curve(labels, scores); J = tpr - fpr; k = np.argmax(J)
    thr_star, sens, spec = float(thr[k]), float(tpr[k]), float(1.0 - fpr[k])
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label='ROC'); plt.plot([0,1],[0,1],'k--', alpha=0.4)
    plt.scatter(1-spec, sens, c='r', label=f"J={sens-(1-spec):.3f}, thr={thr_star:.2f}")
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close()
    return thr_star, sens, spec

def save_confusions_before_after(labels, scores_before, scores_after, thr, fname):
    pb = (scores_before >= thr).astype(int); pa = (scores_after >= thr).astype(int)
    cmb = confusion_matrix(labels, pb); cma = confusion_matrix(labels, pa)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ConfusionMatrixDisplay(cmb).plot(ax=ax[0], colorbar=False)
    ax[0].set_title('Before: single-channel CBF (unsalted)')
    ConfusionMatrixDisplay(cma).plot(ax=ax[1], colorbar=False)
    ax[1].set_title('After: multi-encoding CBF (unsalted + salted)')
    plt.tight_layout(); plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    print("=== Siamese CBF (multi-encoding) for unstructured linkage ===")

    # Paths
    train_path = r"C:\Users\afzal\Documents\GitHub\PACE-COMP3850-Group52\target_train.csv"
    test_path  = r"C:\Users\afzal\Documents\GitHub\PACE-COMP3850-Group52\target_test.csv"

    # Hyperparameters & channels
    bf_len, num_hash, q = 1000, 10, 2
    add_binary_channel = False     # optional extra channel
    extra_q_channel    = None      # e.g., 3 for q=3 extra unsalted+salted channels
    salt2 = "salt-v1"
    batch_size, max_epochs, patience = 64, 40, 5
    use_cosine_head = False

    # Encoder and input size
    encoder = CountingBloomFilter(bf_len=bf_len, num_hash_func=num_hash, q=q, max_count_cap=5)
    base_ch = 2  # unsalted + salted
    bin_ch  = 1 if add_binary_channel else 0
    extra_ch = 2 if extra_q_channel is not None else 0
    total_channels = base_ch + bin_ch + extra_ch
    input_dim = bf_len * total_channels

    # Build model
    siam = SiameseCBF(input_dim=input_dim, embedding_dim=128, l2=1e-4, dropout=0.3, use_cosine_head=use_cosine_head)
    model = siam.build()
    model.summary()

    # Load and parse
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    trL, trR, y_tr = build_pair_texts(train_df, label_col="label")
    teL, teR, y_te = build_pair_texts(test_df,  label_col="label")

    # Encode
    print("Encoding training/test with CBF multi-channels...")
    X1_tr, X2_tr = encode_pairs_cbf(trL, trR, encoder, salt2=salt2,
                                    add_binary=add_binary_channel,
                                    extra_q=extra_q_channel)
    X1_te, X2_te = encode_pairs_cbf(teL, teR, encoder, salt2=salt2,
                                    add_binary=add_binary_channel,
                                    extra_q=extra_q_channel)

    # Class weights
    cls = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = {0: cls[0], 1: cls[1]}

    # Callbacks
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]

    # Train
    history = model.fit([X1_tr, X2_tr], y_tr,
                        validation_data=([X1_te, X2_te], y_te),
                        epochs=max_epochs, batch_size=batch_size,
                        class_weight=class_weights,
                        callbacks=cbs, verbose=1)

    # Predict & tune threshold (TEST here; use VAL if you split)
    y_prob = model.predict([X1_te, X2_te], batch_size=1024).reshape(-1)
    thr_star, f1_star = tune_threshold_f1(y_te, y_prob)
    y_pred = (y_prob >= thr_star).astype(int)
    print(f"Chosen threshold (F1-tuned): {thr_star:.2f} | F1 at thr: {f1_star:.4f}")
    print(classification_report(y_te, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_te, y_pred))

    # Save model
    model.save("siamese_cbf_multi_encoding.keras")  # modern format

    # Training curves
    plt.figure(figsize=(8,4))
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.title("Training Curves")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("S-01_TS-Train_Rule-MultiCBF_2025-10-24.png", dpi=300, bbox_inches="tight")
    plt.close()

    # === NEW: Evidence figures ===
    # S-03 Boundary density (use test here; use val if you have it)
    save_boundary_density(
        scores=y_prob, labels=y_te, thr=thr_star,
        title='Boundary Density (Siamese+Multi-CBF, tuned thr)',
        fname='S-03_TS-ε=Test_Rule-Boundary_2025-10-24.png'
    )

    # S-02 ROC with Youden’s J
    thr_j, sens, spec = save_roc_with_youden(
        scores=y_prob, labels=y_te,
        title="ROC with Youden's J (Siamese+Multi-CBF)",
        fname='S-02_TS-ε=Test_Rule-Thr-F1_2025-10-24.png'
    )
    print(f"Youden J ~ thr={thr_j:.2f}, sens={sens:.3f}, spec={spec:.3f}")

    # S-04 Before/After defence at same threshold:
    # "Before" = single-channel unsalted packed into the same input shape.
    X1_b, X2_b = encode_pairs_cbf_single_channel(teL, teR, encoder, total_channels=total_channels)
    y_prob_before = model.predict([X1_b, X2_b], batch_size=1024).reshape(-1)
    save_confusions_before_after(
        labels=y_te,
        scores_before=y_prob_before,
        scores_after=y_prob,
        thr=thr_star,
        fname='S-04_TS-ε=Test_Rule-Defence-Salt_2025-10-24.png'
    )

    # S-05 Final test confusion matrix (at tuned thr)
    cm = confusion_matrix(y_te, (y_prob >= thr_star).astype(int))
    ConfusionMatrixDisplay(cm).plot(colorbar=False)
    plt.title('Test Confusion Matrix (tuned thr)')
    plt.tight_layout()
    plt.savefig('S-05_TS-ε=Test_Rule-Operate_2025-10-24.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Cleanup
    del X1_tr, X2_tr, X1_te, X2_te, X1_b, X2_b; gc.collect()

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
