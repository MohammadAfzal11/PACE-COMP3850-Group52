# simhash_evaluation.py
import pandas as pd
from private_simhash_linkage import PrivateSimHashLinkage

def run_psh_lsh(A_df, B_df, gt_pairs, eps_list=(0.5, 1.0, 2.0, 5.0), n_bits=256, bands=32, text_col=None):
    """
    Run Private SimHash + LSH with different eps values.
    Returns list of results: [{"eps": ε, "precision": p, "recall": r, "f1": f1}, ...]
    """
    results = []
    n_label = min(100, len(gt_pairs)) if len(gt_pairs) else 0
    label_subset = gt_pairs[:n_label]
    gt_set = set(gt_pairs)

    for eps in eps_list:
        model = PrivateSimHashLinkage(n_bits=n_bits, bands=bands, eps=eps, text_col=text_col)
        info = model.fit(A_df, B_df, label_subset)
        preds = model.link()
        pred_pairs = {(i, j) for (i, j, _) in preds}

        tp = len(pred_pairs & gt_set)
        fp = len(pred_pairs - gt_set)
        fn = len(gt_set - pred_pairs)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0

        results.append({
            "eps": float(eps),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_pairs": len(preds),
            "optimal_threshold": info["optimal_threshold"]
        })

    return results


if __name__ == "__main__":
    # Load your Alice/Bob CSVs
    A = pd.read_csv("Alice_numrec_100_corr_50.csv")
    B = pd.read_csv("Bob_numrec_100_corr_50.csv")

    # Build ground-truth pairs by shared ID if present
    gt = []
    if "id" in A.columns and "id" in B.columns:
        idxB = {v: j for j, v in enumerate(B["id"])}
        for i, v in enumerate(A["id"]):
            if v in idxB:
                gt.append((i, idxB[v]))

    # Fallback: if no IDs, assume row-aligned pairs for testing
    if not gt:
        n = min(len(A), len(B))
        gt = [(i, i) for i in range(n)]

    print("GT pairs:", len(gt))

    # Run evaluation
    results = run_psh_lsh(A, B, gt, eps_list=[0.5, 1.0, 2.0, 5.0])
    print("SimHash + LSH Results:")
    for r in results:
        print(r)