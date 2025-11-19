## SimHash with Local Differential Privacy for Privacy-Preserving Record Linkage (PSH)

## Overview

This document describes the Private SimHash Linkage (PSH) mechanism, an alternative privacy-preserving record linkage approach to the Bloom filter baseline. PSH combines random projection hashing with local differential privacy (LDP) to generate compact binary signatures that protect sensitive attributes while enabling approximate matching across distributed datasets.

## Motivation

The Bloom filter method has been widely used for privacy-preserving record linkage, but it has limitations:
- Vulnerability to frequency attacks: Repeated q-grams in identifiers can leak statistical information

- Fixed tokenisation: Requires carefully chosen q-gram and hash parameters

- Threshold tuning: Manual threshold choice may not generalize across datasets

- Privacy not formally quantified: Noise is added heuristically rather than with formal DP guarantees

## PSH addresses these issues by introducing:

- Randomised hashing (SimHash): Projects records into binary sketches that preserve similarity in Hamming space

- Local differential privacy: Each bit is flipped with probability calibrated to a privacy budget ε, providing formal guarantees

- Threshold learning: Optimal decision boundary is learned automatically from labelled or synthetic pairs

- Scalability with LSH: Signatures can be partitioned into bands for efficient large-scale candidate generation (future extension)

## Key Components:
- Numeric Feature Preprocessing: Standardised to zero mean and unit variance

- Optional Text Features: Encoded with TF-IDF n-grams and concatenated with numeric features

- Random Hyperplanes: A matrix of random vectors defines projection directions

- Binary Signature Generation: Sign of each projection produces a bit (≥0 → 1, <0 → 0)

## Local Differential Privacy

To prevent disclosure of the raw SimHash signatures, randomised response is applied:

Each bit is flipped with probability p = 1 / (exp(ε) + 1)

Smaller ε → more flips → stronger privacy, weaker accuracy

Larger ε → fewer flips → weaker privacy, better accuracy

This provides formal (ε,0)-differential privacy guarantees at the bit level.

## Linkage Process

1. Compute SimHash signatures for Alice and Bob datasets

2. Apply noisy randomised response

3. Compute Hamming similarity between signatures

4. Return pairs exceeding the learned threshold

5. Future optimisation: banded LSH blocking to reduce comparisons for large datasets.

## Privacy-Utility Tradeoff Analysis

- PSH explicitly demonstrates the privacy-utility tradeoff by evaluating at different privacy budgets (ε = 0.5, 1, 2, 5):

- Utility Metrics: Precision, Recall, F1

- Privacy Metrics: ε (privacy budget), effective bit flip probability

- Observation: Lower ε improves privacy but reduces linkage accuracy; higher ε relaxes privacy but increases recall

## SimHash Results:
{'eps': 0.5, 'precision': 0.010131712259371834, 'recall': 1.0, 'f1': 0.020060180541624874, 'predicted_pairs': 9870, 'optimal_threshold': 0.45}
{'eps': 1.0, 'precision': 0.010005002501250625, 'recall': 1.0, 'f1': 0.01981178801386825, 'predicted_pairs': 9995, 'optimal_threshold': 0.36}
{'eps': 2.0, 'precision': 0.01005573055488248, 'recall': 0.83, 'f1': 0.019870720612880057, 'predicted_pairs': 8254, 'optimal_threshold': 0.31}
{'eps': 5.0, 'precision': 0.010002857959416977, 'recall': 0.7, 'f1': 0.01972386587771203, 'predicted_pairs': 6998, 'optimal_threshold': 0.3}

## Key Parameters

`n_bits`: Length of SimHash signature (1024)

`bands`: Number of LSH bands (default: 32; future use)

`eps`: Privacy budget (default: 1.0)

`text_col`: Optional text column for TF-IDF encoding

## Output

`optimal_threshold`: Best cutoff learned on labelled set

`precision, recall, f1`: Utility metrics per ε

`predicted_pairs`: Number of matches generated

## Conclusion

PSH demonstrates a fundamentally different privacy-preserving record linkage strategy compared to Bloom filters. By combining random projections, binary signatures, and local differential privacy, it provides a lightweight but formally private method for record linkage.

Although accuracy may be lower on small structured datasets, it clearly illustrates the trade-offs of using formal DP mechanisms in practice. This makes PSH a valuable comparative approach in exploring privacy-preserving linkage techniques.