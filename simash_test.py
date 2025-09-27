import pandas as pd
from private_simhash_linkage import PrivateSimHashLinkage

# Example toy data
A = pd.read_csv("Alice_numrec_100_corr_50.csv")
B = pd.read_csv("Bob_numrec_100_corr_50.csv")

# Build a tiny ground-truth list of index matches for threshold learning.
# If both files share a hidden "id" you can map it; otherwise hand-label a few for now.
gt = []
if "id" in A.columns and "id" in B.columns:
    idxB = {v:i for i,v in enumerate(B["id"])}
    for i, v in enumerate(A["id"]):
        if v in idxB:
            gt.append((i, idxB[v]))

psh = PrivateSimHashLinkage(n_bits=256, bands=32, eps=1.0, text_col=None)
info = psh.fit(A, B, gt[:50])            # use a small labelled subset to learn threshold
pairs = psh.link()                        # [(i, j, sim), ...] sorted by sim desc
print(info, pairs[:10])