# private_simhash_linkage.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class PrivateSimHashLinkage:
    def __init__(self, n_bits=256, bands=32, eps=1.0, text_col=None, random_state=42):
        self.n_bits = n_bits
        self.bands = bands
        self.r = n_bits // bands
        self.eps = eps
        self.text_col = text_col
        self.rs = np.random.RandomState(random_state)
        self.W = None
        self.scaler = None
        self.vectorizer = None
        self.threshold = 0.5

# length of the binary signature; higher = more discriminative but slower
#bands=32,          # number of LSH bands (used when you add banded blocking); n_bits must be divisible by bands
#eps=1.0,           # local-DP strength (ε). Lower ε = stronger privacy = more bit flips 
#text_col=None,     # optional text column name to vectorize (TF-IDF); if None, only numeric features used
#random_state=42):  # reproducibility for projections and DP flips

#EACH SECTION Includes:
# 1. _prep(df): Splits out text (if any), scaled numeric  features, TF-IDFs, concatenates into a single feature vector
# 2. _simhash(X): builds random hyperplanes once, sign projections into bits then applies randomise responses to each bit
# 3. _similarity(a, b): 1 - Hamming(a, b) / n_bits in [0, 1]
# 4. fit(A ,B, matches): learns the decision threshold that maximises F1 on a small
# 5. link(): return a list of (i, j, similarity) for pairs above the learned threshold

# DEVELOPER NOTES: link() is brute forced, branded LSH can be later added


    def _prep(self, df):
        X_struct = df.drop(columns=[self.text_col]) if self.text_col and self.text_col in df else df
        X_struct = X_struct.select_dtypes(exclude=['object']).fillna(0.0).values
        if self.scaler is None: self.scaler = StandardScaler().fit(X_struct)
        Xs = self.scaler.transform(X_struct)
        if self.text_col and self.text_col in df:
            texts = df[self.text_col].astype(str).fillna("").tolist()
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
                Xt = self.vectorizer.fit_transform(texts).toarray()
            else:
                Xt = self.vectorizer.transform(texts).toarray()
            X = np.hstack([Xs, Xt])
        else:
            X = Xs
        return X
    

    def _simhash(self, X):
        d = X.shape[1]
        if self.W is None:
            self.W = self.rs.normal(0, 1, size=(self.n_bits, d))
        proj = X @ self.W.T
        bits = (proj >= 0).astype(np.int8)
        # Local DP via randomized response
        p = 1.0 / (np.exp(self.eps) + 1.0)
        flips = self.rs.binomial(1, p, size=bits.shape).astype(bool)
        bits = np.where(flips, 1 - bits, bits)
        return bits

    def _similarity(self, a, b):
        # 1 - normalized Hamming distance
        return 1.0 - (np.bitwise_xor(a, b).sum(axis=1) / self.n_bits)
    
    def fit(self, A: pd.DataFrame, B: pd.DataFrame, matches):
        XA, XB = self._prep(A), self._prep(B)
        SA, SB = self._simhash(XA), self._simhash(XB)
        # learn threshold from labelled pairs
        sims, y = [], []
        for i, j in matches:
            sims.append(self._similarity(SA[i:i+1], SB[j:j+1])[0])
            y.append(1)
        # add some negatives
        import random
        nneg = len(matches)*3
        for _ in range(nneg):
            i = random.randrange(len(SA)); j = random.randrange(len(SB))
            if (i, j) not in matches:
                sims.append(self._similarity(SA[i:i+1], SB[j:j+1])[0]); y.append(0)
        sims, y = np.array(sims), np.array(y)
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.3, 0.9, 61):
            pred = (sims >= t).astype(int)
            tp = ((pred==1)&(y==1)).sum(); fp = ((pred==1)&(y==0)).sum(); fn = ((pred==0)&(y==1)).sum()
            precision = tp/(tp+fp) if tp+fp>0 else 0.0
            recall    = tp/(tp+fn) if tp+fn>0 else 0.0
            f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
            if f1>best_f1: best_f1, best_t = f1, t
        self.threshold = best_t
        self.SA, self.SB = SA, SB
        return {'optimal_threshold': self.threshold, 'epsilon': self.eps} 

    def link(self):
        matches = []
        for i in range(len(self.SA)):
            sims = 1.0 - (np.bitwise_xor(self.SA[i], self.SB).sum(axis=1) / self.n_bits)
            for j, s in enumerate(sims):
                if s >= self.threshold:
                    matches.append((i, j, float(s)))
        return sorted(matches, key=lambda x: x[2], reverse=True)       