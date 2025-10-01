# Code Explanation: Privacy-Preserving Record Linkage Models

This document explains the implementation and functionality of both PPRL and FPN-RL models for privacy-preserving record linkage.

---

## Table of Contents
1. [PPRL Model (Bloom Filter-Based)](#pprl-model)
2. [FPN-RL Model (Neural Network-Based)](#fpn-rl-model)
3. [Parameter Comparison Framework](#parameter-comparison)
4. [Performance Metrics](#performance-metrics)

---

## PPRL Model (Bloom Filter-Based) {#pprl-model}

### Overview
The PPRL (Privacy-Preserving Record Linkage) model uses Bloom filters with differential privacy to match records while protecting sensitive information. It encodes quasi-identifiers (QIDs) like names and addresses into bit vectors, adds calibrated noise for privacy, and uses similarity-based matching.

### CHANGE 1: Import Dependencies
```python
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from BF import BF
from PPRL import Link
```

**What it does:**
- Imports core libraries for data processing, visualization, and the PPRL implementation
- `BF`: Bloom Filter implementation module
- `Link`: Main PPRL linkage class with privacy mechanisms

---

### CHANGE 2: Configure Parameters for Linkage

```python
BF_length = 1000        # Bloom filter length
BF_num_hash = 10        # Number of hash functions
BF_q_gram = 2           # Q-gram size for tokenization
min_sim_val = 0.8       # Similarity threshold for matching
link_attrs = [1,2,3,4]  # Attributes to use for linkage
block_attrs = [2,4]     # Attributes to use for blocking
ent_index = 0           # Entity ID column index
epsilon = 7             # Privacy budget
```

**What it does:**
- **BF_length**: Controls Bloom filter size. Larger values reduce collision probability but increase memory usage.
- **BF_num_hash**: Number of hash functions used. More hashes reduce false positives but increase computation time.
- **BF_q_gram**: Size of character n-grams for tokenization. Typically 2 or 3.
- **min_sim_val**: Threshold for classifying record pairs as matches. Higher values mean stricter matching.
- **epsilon**: Privacy budget parameter. Lower values provide stronger privacy but reduce accuracy.

**How to tune for better accuracy:**
1. Increase `epsilon` (e.g., 10-15) for higher accuracy but less privacy
2. Decrease `min_sim_val` (e.g., 0.7-0.75) to capture more matches but risk false positives
3. Increase `BF_length` (e.g., 1500-2000) to reduce hash collisions
4. Adjust `BF_num_hash` (e.g., 8-12) to balance false positive rate

---

### CHANGE 3: Load Datasets

```python
dataset1_path = '../csv_files/Alice_numrec_100_corr_50.csv'
dataset2_path = '../csv_files/Bob_numrec_100_corr_50.csv'

db1 = link.read_database(dataset1_path)
db2 = link.read_database(dataset2_path)
```

**What it does:**
- Loads two datasets for record linkage
- Datasets contain voter registration data with fields like `first_name`, `last_name`, `city`, etc.
- The number (e.g., 100, 500) indicates record count
- The correlation (e.g., corr_25, corr_50) indicates the percentage of matching records between datasets

**Dataset variations:**
- `100_corr_25`: 100 records, 25% overlap
- `100_corr_50`: 100 records, 50% overlap
- `500_corr_25`: 500 records, 25% overlap
- `500_corr_50`: 500 records, 50% overlap

---

### CHANGE 4: PPRL Workflow

```python
blk_ind1 = link.build_BI(db1)
blk_ind2 = link.build_BI(db2)
```

**Blocking:**
- Groups records into blocks based on blocking attributes (e.g., city, last name)
- Reduces comparison space from O(n²) to O(n)
- Only records in the same block are compared

```python
bf_dict1, all_val_set1 = link.data_encode(db1)
bf_dict2, all_val_set2 = link.data_encode(db2)
```

**Bloom Filter Encoding:**
- Converts record attributes into Bloom filter bit vectors
- Each attribute value is tokenized into q-grams
- Q-grams are hashed and set corresponding bits in the Bloom filter
- Example: "John" with q=2 → ["Jo", "oh", "hn"] → hash to bit positions

```python
fpr = (1 - math.e**((-1*BF_num_hash*num_total_all_val_set)/BF_length))**BF_num_hash
```

**False Positive Rate (FPR):**
- Calculates theoretical FPR of Bloom filters
- Lower FPR indicates better encoding quality
- Formula: (1 - e^(-k*n/m))^k where k=num_hash, n=num_elements, m=BF_length

```python
bf_dict1_dp = link.dp_bloom_filters(bf_dict1)
bf_dict2_dp = link.dp_bloom_filters(bf_dict2)
```

**Differential Privacy:**
- Adds calibrated Laplace noise to Bloom filter bit values
- Noise scale is determined by epsilon parameter
- Provides (ε, 0)-differential privacy guarantee
- Lower epsilon = more noise = stronger privacy but lower accuracy

```python
matches = link.match(blk_ind1, blk_ind2, bf_dict1_dp, bf_dict2_dp)
```

**Matching:**
- Compares Bloom filters using Dice coefficient similarity
- Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
- Pairs with similarity > min_sim_val are classified as matches

---

### CHANGE 5: Evaluate PPRL Performance

```python
prec, rec, f1 = link.evaluate(matches, db1, db2)
```

**Evaluation Metrics:**
- **Precision**: Proportion of predicted matches that are true matches
  - Formula: TP / (TP + FP)
- **Recall**: Proportion of true matches that were found
  - Formula: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)

**What good results look like:**
- Precision ≥ 0.9: Very few false positives
- Recall ≥ 0.7: Most true matches are found
- F1 ≥ 0.8: Good overall balance

---

### CHANGE 6: Baseline Comparisons

```python
matches_npp = link.match_npp(blk_ind1, blk_ind2, db1, db2)
```

**Non-Privacy-Preserving Baseline:**
- Matches records using raw attribute values (no encoding, no privacy)
- Provides upper bound on linkage quality
- Shows utility cost of privacy protection

```python
matches_nodp = link.match(blk_ind1, blk_ind2, bf_dict1, bf_dict2)
```

**PPRL without Differential Privacy:**
- Uses Bloom filters but no differential privacy noise
- Shows impact of DP noise on linkage quality
- Provides "probable" privacy through Bloom filters but no formal guarantees

---

### CHANGE 7: Parameter Comparison Experiments

```python
def test_parameter_variations(datasets, param_name, param_values, base_params):
```

**What it does:**
- Systematically tests different parameter values across multiple datasets
- Runs complete PPRL workflow for each configuration
- Collects precision, recall, and F1 scores
- Returns results as a pandas DataFrame for analysis

**Process for each combination:**
1. Create Link instance with modified parameter
2. Load and preprocess datasets
3. Apply blocking
4. Encode to Bloom filters
5. Add differential privacy noise
6. Perform matching
7. Evaluate and store results

---

### CHANGE 8: Privacy Budget (Epsilon) Analysis

```python
epsilon_values = [1, 3, 5, 7, 10, 15]
epsilon_results = test_parameter_variations(datasets, 'epsilon', epsilon_values, base_params)
```

**What it tests:**
- How privacy budget affects linkage accuracy across different datasets
- Lower epsilon = stronger privacy but lower accuracy
- Higher epsilon = weaker privacy but higher accuracy

**Generates three plots:**
1. Precision vs Epsilon: Shows how matching precision changes with privacy level
2. Recall vs Epsilon: Shows how match detection rate changes with privacy level
3. F1 Score vs Epsilon: Shows overall performance across privacy levels

**Expected patterns:**
- F1 score increases with epsilon (less noise, better matching)
- Larger datasets (500 records) typically perform better
- Higher correlation datasets (corr_50) show better results

---

### CHANGE 9: Similarity Threshold Analysis

```python
threshold_values = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
threshold_results = test_parameter_variations(datasets, 'min_sim_val', threshold_values, base_params)
```

**What it tests:**
- How matching threshold affects precision/recall tradeoff
- Lower threshold = more matches (higher recall, lower precision)
- Higher threshold = fewer matches (lower recall, higher precision)

**Expected patterns:**
- Precision increases with threshold (fewer false positives)
- Recall decreases with threshold (more false negatives)
- Optimal F1 typically at threshold 0.75-0.8

---

### CHANGE 10: Bloom Filter Length Analysis

```python
bf_length_values = [500, 1000, 1500, 2000]
bf_length_results = test_parameter_variations(datasets, 'BF_length', bf_length_values, base_params)
```

**What it tests:**
- How Bloom filter size affects collision probability and accuracy
- Larger BF_length = fewer collisions = better distinguishability
- But also more memory and computation

**Expected patterns:**
- Performance improves with BF length up to a point
- Diminishing returns after optimal size (typically 1000-1500 for these datasets)
- Smaller datasets benefit less from larger Bloom filters

---

### CHANGE 11: Number of Hash Functions Analysis

```python
num_hash_values = [5, 10, 15, 20]
num_hash_results = test_parameter_variations(datasets, 'BF_num_hash', num_hash_values, base_params)
```

**What it tests:**
- How number of hash functions affects false positive rate
- More hashes = lower FPR but more computation
- Optimal value depends on BF_length and data density

**Expected patterns:**
- Optimal typically around 8-12 hash functions
- Too few hashes = high collision rate
- Too many hashes = over-saturation of Bloom filter

---

### CHANGE 12: Combined Performance Comparison

**What it does:**
- Creates a 2x2 grid showing all four parameter analyses side-by-side
- Allows visual comparison of which parameters have most impact
- Shows dataset-specific performance patterns

**Key insights from combined view:**
1. Epsilon has largest impact on performance (privacy-utility tradeoff)
2. Threshold controls precision/recall balance
3. BF_length and num_hash have smaller but important effects
4. Larger datasets (500) consistently outperform smaller ones (100)
5. Higher correlation datasets (corr_50) easier to link than lower (corr_25)

---

## FPN-RL Model (Neural Network-Based) {#fpn-rl-model}

### Overview
The FPN-RL (Federated Privacy-Preserving Neural Network Record Linkage) model uses deep learning to learn privacy-preserving embeddings for record linkage. It combines neural networks, federated learning principles, and differential privacy.

---

### CHANGE 1: Import Dependencies

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
```

**What it does:**
- TensorFlow/Keras for building and training neural networks
- NumPy for numerical operations
- Pandas for data manipulation

---

### CHANGE 2: FPN-RL Model Definition

```python
class FederatedEmbeddingLinkage:
```

**Core components:**
1. **Encoder Model**: Neural network that maps records to embeddings
2. **Privacy Mechanism**: Adds Gaussian noise to embeddings for differential privacy
3. **Similarity Computation**: Calculates cosine similarity between embeddings
4. **Threshold Learning**: Adaptive threshold for match/non-match classification

**Key innovations:**
- Works with both structured (names, addresses) and unstructured (text descriptions) data
- TF-IDF vectorization for text features
- Mixed-mode processing for heterogeneous data

---

### CHANGE 3: Privacy-Preserving Methods

```python
def _add_differential_privacy_noise(self, embeddings: np.ndarray) -> np.ndarray:
```

**What it does:**
- Adds calibrated Gaussian noise to learned embeddings
- Noise scale: σ = (2 * L2_clip * noise_multiplier) / epsilon
- Provides (ε, δ)-differential privacy guarantee
- L2 norm clipping prevents sensitivity amplification

**Privacy mechanism:**
1. Clip embedding L2 norm to bound sensitivity
2. Calculate noise scale based on epsilon
3. Add Gaussian noise N(0, σ²)
4. Result: private embeddings that preserve utility while protecting privacy

---

### CHANGE 4: Neural Network Architecture

```python
def _build_encoder_model(self, input_dim: int):
```

**Network structure:**
```
Input (variable dim) 
  ↓
Dense(256, ReLU) + Dropout(0.3)
  ↓
Dense(128, ReLU) + Dropout(0.3)
  ↓
Dense(embedding_dim) + L2 Normalization
  ↓
Embedding output
```

**Key features:**
- Multiple dense layers with ReLU activation
- Dropout for regularization (prevents overfitting)
- L2 regularization on weights
- Final L2 normalization for stable embeddings
- Embedding dimension controls representation capacity

---

### CHANGE 5: Record Linkage Methods

```python
def train_and_link(self, data1, data2, ground_truth, epochs=50, batch_size=32):
```

**Training process:**
1. **Data preprocessing**: Standardize numerical features, encode categorical
2. **Feature extraction**: TF-IDF for text, one-hot for categorical
3. **Pair generation**: Create positive (matching) and negative (non-matching) pairs
4. **Training**: Siamese network with contrastive loss
5. **Embedding generation**: Encode both datasets
6. **Privacy application**: Add differential privacy noise
7. **Matching**: Compute similarities and apply threshold
8. **Evaluation**: Calculate precision, recall, F1

**Contrastive loss:**
- Minimizes distance for matching pairs
- Maximizes distance for non-matching pairs
- Margin-based: encourages clear separation

---

### CHANGE 6: Configure FPN-RL Parameters

```python
embedding_dim = 64          # Embedding dimension
epsilon = 1.0               # Privacy budget
delta = 1e-5                # Privacy parameter
min_sim_threshold = 0.7     # Similarity threshold
learning_rate = 0.001       # Learning rate
epochs = 50                 # Training epochs
batch_size = 32             # Batch size
```

**Parameter meanings:**

**embedding_dim (32-256):**
- Dimensionality of learned representations
- Higher = more capacity but risk of overfitting
- Recommended: 64-128 for these datasets

**epsilon (0.5-10):**
- Privacy budget (same as PPRL but applied to embeddings)
- Lower = stronger privacy but noisier embeddings
- Recommended: 1.0-5.0 for good privacy-utility balance

**delta (1e-7 to 1e-5):**
- Probability of privacy failure
- Very small value (cryptographic security)
- Typically fixed at 1e-5

**min_sim_threshold (0.5-0.9):**
- Classification boundary for matches
- Lower = more matches (higher recall, lower precision)
- Recommended: 0.6-0.8

**learning_rate (0.0001-0.01):**
- Step size for gradient descent
- Too high = unstable training
- Too low = slow convergence
- Recommended: 0.001-0.005

**epochs (10-100):**
- Number of training passes through data
- More epochs = better convergence but longer training
- Use early stopping if validation loss plateaus
- Recommended: 30-70 epochs

**batch_size (16-64):**
- Number of samples per gradient update
- Larger = faster training but more memory
- Smaller = noisier gradients but better generalization
- Recommended: 32

---

### CHANGE 7: Sample Data Generation

```python
def generate_sample_data_with_text(n_records=100, match_rate=0.3):
```

**What it does:**
- Generates synthetic datasets with controlled match rate
- Creates records with names, ages, cities, and text descriptions
- Introduces realistic variations (typos, abbreviations, missing data)
- Returns ground truth matches for evaluation

---

### CHANGE 8: Generate Test Data

```python
data1, data2, ground_truth = generate_sample_data_with_text(n_records=50, match_rate=0.4)
```

**Purpose:**
- Creates test datasets for quick experimentation
- Controllable parameters for difficulty
- Known ground truth for accurate evaluation

---

### CHANGE 9: Load CSV Datasets

```python
csv_data1 = pd.read_csv('../csv_files/Alice_numrec_100_corr_50.csv')
csv_data2 = pd.read_csv('../csv_files/Bob_numrec_100_corr_50.csv')
```

**What it does:**
- Loads real voter registration datasets
- Same datasets used by PPRL for fair comparison
- Requires preprocessing to add text features for FPN-RL

---

### CHANGE 10: Train FPN-RL Model

```python
results = fpn_rl.train_and_link(
    data1, data2, ground_truth,
    epochs=epochs,
    batch_size=batch_size
)
```

**Training workflow:**
1. Extract and preprocess features
2. Build encoder model
3. Generate training pairs (positive and negative)
4. Train with contrastive loss
5. Generate embeddings for both datasets
6. Add differential privacy noise
7. Compute pairwise similarities
8. Apply threshold for classification
9. Evaluate against ground truth

**Output:**
- Training loss convergence
- Final precision, recall, F1 scores
- Privacy budget consumption

---

### CHANGE 11: Parameter Comparison Experiments

```python
def test_fpn_rl_parameter_variations(param_name, param_values, base_params, datasets_dict):
```

**What it does:**
- Similar to PPRL but for neural network parameters
- Tests each parameter across multiple datasets
- Computationally intensive (neural network training)
- Returns comprehensive performance metrics

---

### CHANGE 12: Privacy Budget (Epsilon) Analysis

```python
epsilon_values_fpn = [0.5, 1.0, 2.0, 5.0, 10.0]
```

**What it tests:**
- Impact of privacy noise on learned embeddings
- FPN-RL typically more robust to noise than PPRL
- Neural network learns to be noise-resistant

**Expected patterns:**
- Less dramatic drop in accuracy compared to PPRL
- Performance stable even at lower epsilon (0.5-1.0)
- Embeddings capture semantic information better than Bloom filters

---

### CHANGE 13: Embedding Dimension Analysis

```python
embedding_dim_values = [32, 64, 128, 256]
```

**What it tests:**
- Model capacity vs. overfitting tradeoff
- Computational cost increases with dimension
- Diminishing returns after optimal size

**Expected patterns:**
- Performance improves from 32 to 64
- Plateau or slight improvement from 64 to 128
- Possible overfitting at 256 for small datasets
- Larger datasets benefit more from higher dimensions

---

### CHANGE 14: Similarity Threshold Analysis

```python
threshold_values_fpn = [0.5, 0.6, 0.7, 0.8, 0.9]
```

**What it tests:**
- Precision/recall tradeoff in embedding space
- Optimal threshold depends on embedding quality
- Similar to PPRL but typically lower optimal threshold

**Expected patterns:**
- Optimal F1 at threshold 0.6-0.7 (lower than PPRL's 0.75-0.8)
- Embeddings have better separation than Bloom filter similarities
- More pronounced precision/recall tradeoff

---

### CHANGE 15: Combined Performance Comparison

**What it shows:**
- Side-by-side comparison of all parameters
- 3 subplots: epsilon, embedding_dim, threshold
- Dataset-specific patterns clearly visible

**Key insights:**
1. Epsilon has moderate impact (less than PPRL)
2. Embedding dimension important for model capacity
3. Threshold controls final classification
4. FPN-RL generally more robust than PPRL
5. Better performance on complex/noisy data

---

## Parameter Comparison Framework {#parameter-comparison}

### Purpose
The parameter comparison framework systematically evaluates how different hyperparameter settings affect linkage quality across various datasets.

### Methodology

**1. Parameter Selection:**
- Choose one parameter to vary (e.g., epsilon)
- Fix all other parameters at baseline values
- Test multiple values for the chosen parameter

**2. Dataset Selection:**
- Test on 4 datasets with varying sizes and match rates
- 100 records: Smaller, faster testing
- 500 records: More realistic scale
- 25% correlation: Harder matching task
- 50% correlation: Easier matching task

**3. Experiment Execution:**
- For each (parameter value, dataset) combination:
  - Run complete linkage workflow
  - Record precision, recall, F1 score
  - Track computational time

**4. Visualization:**
- Line plots showing performance vs. parameter value
- Separate lines for each dataset
- Multiple metrics (precision, recall, F1) side-by-side

**5. Analysis:**
- Identify optimal parameter ranges
- Compare model behavior across datasets
- Understand privacy-utility tradeoffs

---

## Performance Metrics {#performance-metrics}

### Confusion Matrix

```
                Predicted Match    Predicted Non-Match
Actual Match         TP                   FN
Actual Non-Match     FP                   TN
```

**TP (True Positives):** Correctly identified matches  
**FP (False Positives):** Incorrectly identified matches  
**FN (False Negatives):** Missed matches  
**TN (True Negatives):** Correctly identified non-matches

---

### Precision

```
Precision = TP / (TP + FP)
```

**Interpretation:**
- "Of all predicted matches, how many are correct?"
- High precision = few false alarms
- Important when false matches are costly
- PPRL typically achieves 0.85-1.0
- FPN-RL typically achieves 0.80-0.95

---

### Recall

```
Recall = TP / (TP + FN)
```

**Interpretation:**
- "Of all true matches, how many did we find?"
- High recall = few missed matches
- Important when missing matches is costly
- PPRL typically achieves 0.60-0.85
- FPN-RL typically achieves 0.70-0.90

---

### F1 Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Interpretation:**
- Harmonic mean of precision and recall
- Balances both metrics equally
- Single number for overall performance
- Good F1 ≥ 0.80
- Excellent F1 ≥ 0.90

---

### Privacy Metrics

**For PPRL:**
- **Epsilon (ε):** Privacy budget (lower = stronger privacy)
- **False Positive Rate:** Probability of Bloom filter collision

**For FPN-RL:**
- **Epsilon (ε):** Privacy budget for embedding noise
- **Delta (δ):** Probability of privacy failure

---

## Summary

### PPRL (Bloom Filter-Based)
**Strengths:**
- Fast computation (no training required)
- Provable differential privacy guarantees
- Simple to understand and implement
- Low memory footprint

**Weaknesses:**
- Fixed encoding (no learning)
- Sensitive to parameter choices
- Performance degrades with strong privacy
- Limited to structured data

**Best for:**
- Large-scale deployment
- When interpretability is crucial
- When training data is unavailable
- Real-time matching requirements

---

### FPN-RL (Neural Network-Based)
**Strengths:**
- Learns optimal representations
- Handles mixed data types
- More robust to privacy noise
- Better performance on complex data

**Weaknesses:**
- Requires training data
- Computationally expensive
- Longer development cycle
- Needs hyperparameter tuning

**Best for:**
- Complex/noisy datasets
- When training data is available
- When maximum accuracy is needed
- Research and development scenarios

---

### Parameter Tuning Guidelines

**To maximize accuracy:**
1. Increase epsilon (weaker privacy)
2. Lower similarity threshold
3. Increase Bloom filter length (PPRL)
4. Increase embedding dimension (FPN-RL)

**To maximize privacy:**
1. Decrease epsilon (stronger privacy)
2. Increase similarity threshold
3. Add more noise (accept accuracy loss)

**To balance both:**
1. Use epsilon = 1.0-5.0
2. Use threshold = 0.7-0.8
3. Test on representative datasets
4. Monitor F1 score as primary metric

---

## Visualization Outputs

All experiments generate publication-quality plots saved as PNG files:

**PPRL:**
- `pprl_epsilon_comparison.png`: Privacy budget analysis
- `pprl_threshold_comparison.png`: Threshold analysis
- `pprl_bf_length_comparison.png`: Bloom filter size analysis
- `pprl_num_hash_comparison.png`: Hash function analysis
- `pprl_combined_comparison.png`: All parameters together

**FPN-RL:**
- `fpn_rl_epsilon_comparison.png`: Privacy budget analysis
- `fpn_rl_embedding_dim_comparison.png`: Embedding dimension analysis
- `fpn_rl_threshold_comparison.png`: Threshold analysis
- `fpn_rl_combined_comparison.png`: All parameters together

Each plot shows:
- X-axis: Parameter value being tested
- Y-axis: Performance metric (precision/recall/F1)
- Lines: Different datasets
- Grid: For easy reading
- Legend: Dataset identification

---

## Running the Code

**For PPRL:**
1. Open `PPRL.ipynb` in Jupyter
2. Modify parameters in CHANGE 2
3. Modify dataset paths in CHANGE 3
4. Run all cells sequentially
5. View results and generated plots

**For FPN-RL:**
1. Open `federated_embedding_linkage.ipynb` in Jupyter
2. Modify parameters in CHANGE 6
3. Modify dataset paths in CHANGE 9
4. Run all cells sequentially
5. View results and generated plots

**Note:** Full parameter comparison experiments can take 10-30 minutes depending on hardware.
