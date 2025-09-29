# Federated Privacy-Preserving Neural Network Record Linkage (FPN-RL)

## Overview

This document describes the **Federated Privacy-Preserving Neural Network Record Linkage (FPN-RL)** mechanism, a novel approach to privacy-preserving record linkage that extends beyond the traditional threshold-based Bloom filter approach used in PPRL.ipynb. FPN-RL introduces a hybrid neural network architecture that can handle both structured and unstructured data while providing strong differential privacy guarantees.

## Motivation

The existing PPRL implementation in this repository uses Bloom filters with threshold-based classification, which has several limitations:

1. **Limited to structured data**: Cannot effectively handle unstructured text data
2. **Fixed similarity metrics**: Uses predefined similarity measures that may not capture complex relationships
3. **Static thresholds**: Requires manual threshold tuning
4. **Privacy-utility tradeoff**: Difficult to analyze and optimize the balance between privacy and utility

FPN-RL addresses these limitations by introducing:

- **Neural embedding learning**: Automatically learns optimal representations for both structured and unstructured data
- **Federated learning principles**: Distributes privacy protection across the learning process
- **Adaptive threshold learning**: Automatically learns optimal decision boundaries
- **Comprehensive privacy-utility analysis**: Provides tools to analyze and optimize privacy-utility tradeoffs

## Technical Architecture

### 1. Federated Embedding Learning

The core innovation is a federated approach to learning privacy-preserving embeddings:

```
Input Data → Feature Preprocessing → Neural Encoder → Private Embeddings → Linkage Classification
```

#### Key Components:

- **Multi-modal Preprocessing**: Handles structured data (categorical, numerical) and unstructured data (text) through unified feature engineering
- **Privacy-Aware Autoencoder**: Learns compressed representations while maintaining differential privacy
- **Embedding Space Privacy**: Applies differential privacy noise at the embedding level rather than the raw data level

### 2. Differential Privacy Guarantees

FPN-RL provides (ε, δ)-differential privacy guarantees through:

- **Gaussian Mechanism**: Adds calibrated Gaussian noise to embeddings
- **L2 Norm Clipping**: Bounds the sensitivity of the embedding function
- **Privacy Accounting**: Tracks privacy budget consumption across training and inference

#### Privacy Parameters:
- `ε (epsilon)`: Privacy budget - smaller values provide stronger privacy
- `δ (delta)`: Failure probability - typically set to 1e-5
- `noise_multiplier`: Controls the amount of noise added
- `l2_norm_clip`: Bounds the L2 norm of gradients/embeddings

### 3. Neural Architecture

#### Encoder Model:
```
Input Features → Dense(256) → BatchNorm → Dropout → 
Dense(128) → BatchNorm → Dropout → 
Dense(embedding_dim) → Tanh Activation
```

#### Classifier Model:
```
Embedding Difference → Dense(64) → BatchNorm → Dropout →
Dense(32) → BatchNorm → Dropout →
Dense(16) → Dropout →
Dense(1) → Sigmoid
```

### 4. Training Process

FPN-RL uses a multi-phase training approach:

#### Phase 1: Unsupervised Pre-training
- Trains an autoencoder to learn meaningful data representations
- Reconstruction loss encourages the model to preserve important information
- Privacy constraints are enforced during this phase

#### Phase 2: Supervised Linkage Learning
- Uses labeled matching pairs to train the classification component
- Learns to distinguish between matching and non-matching record pairs
- Applies differential privacy noise to embeddings

#### Phase 3: Threshold Optimization
- Automatically learns optimal decision thresholds
- Maximizes F1 score on validation data
- Adapts to specific dataset characteristics

## Data Processing Capabilities

### Structured Data Processing

For structured data, FPN-RL employs multiple encoding strategies:

1. **Categorical Data**: 
   - Hash-based encoding with multiple hash functions
   - String similarity features
   - Collision-resistant representation

2. **Numerical Data**:
   - Normalization and standardization
   - Privacy-preserving noise addition
   - Robust to missing values

### Unstructured Data Processing

For text data, FPN-RL uses:

1. **TF-IDF Vectorization**: Converts text to numerical features
2. **N-gram Features**: Captures local text patterns
3. **Privacy-Preserving Text Analysis**: Applies privacy constraints to text embeddings

## Privacy-Utility Tradeoff Analysis

FPN-RL provides comprehensive tools for analyzing the privacy-utility tradeoff:

### Evaluation Metrics

1. **Utility Metrics**:
   - Precision: Fraction of predicted matches that are correct
   - Recall: Fraction of true matches that are detected
   - F1 Score: Harmonic mean of precision and recall

2. **Privacy Metrics**:
   - Privacy budget consumption (ε)
   - Privacy cost (1/ε)
   - Composition accounting

### Tradeoff Analysis

The system enables systematic evaluation across different privacy budgets:

```python
epsilon_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
tradeoff_results = model.evaluate_privacy_utility_tradeoff(
    epsilon_range, test_data1, test_data2, ground_truth_matches
)
```

## Advantages over Threshold-based PPRL

| Feature | Threshold-based PPRL | FPN-RL |
|---------|---------------------|---------|
| Data Types | Structured only | Structured + Unstructured |
| Privacy Mechanism | Bloom filters + DP noise | Neural embeddings + DP |
| Threshold Learning | Manual tuning | Automatic optimization |
| Scalability | Limited by blocking | Scalable neural architecture |
| Feature Learning | Hand-crafted similarities | Learned representations |
| Privacy Analysis | Basic privacy guarantees | Comprehensive tradeoff analysis |

## Implementation Details

### Key Parameters

- `embedding_dim`: Dimension of learned embeddings (default: 128)
- `epsilon`: Differential privacy budget (default: 1.0)
- `delta`: DP failure probability (default: 1e-5)
- `noise_multiplier`: Gaussian noise scaling (default: 1.1)
- `min_sim_threshold`: Minimum similarity threshold (default: 0.5)

### Performance Considerations

1. **Memory Usage**: Neural models require more memory than Bloom filters
2. **Computation Time**: Training is more intensive but enables better accuracy
3. **Scalability**: Can handle larger datasets through batch processing

## Usage Example

```python
from federated_embedding_linkage import FederatedEmbeddingLinkage

# Initialize the model
model = FederatedEmbeddingLinkage(
    embedding_dim=128,
    epsilon=1.0,
    delta=1e-5
)

# Train on datasets with both structured and text data
training_results = model.train(
    data1=alice_data,
    data2=bob_data,
    ground_truth_matches=known_matches,
    text_col='description',  # Column containing unstructured text
    epochs=100
)

# Perform record linkage
matches = model.link_records(
    data1=test_alice,
    data2=test_bob,
    text_col='description'
)

# Analyze privacy-utility tradeoff
tradeoff_results = model.evaluate_privacy_utility_tradeoff(
    epsilon_range=[0.1, 0.5, 1.0, 2.0, 5.0],
    test_data1=test_alice,
    test_data2=test_bob,
    ground_truth_matches=test_matches
)

# Plot results
model.plot_privacy_utility_tradeoff(tradeoff_results)
```

## Comparison with PPRL.ipynb Results

To compare FPN-RL with the existing threshold-based PPRL approach:

1. **Load the same datasets** used in PPRL.ipynb
2. **Train both models** with identical privacy budgets
3. **Compare metrics**: Precision, Recall, F1 Score
4. **Analyze privacy guarantees**: Both provide differential privacy but through different mechanisms
5. **Evaluate scalability**: FPN-RL can handle larger, more complex datasets

## Future Extensions

1. **Multi-party Linkage**: Extend to link records across more than two databases
2. **Blockchain Integration**: Use blockchain for federated privacy guarantees
3. **Advanced Privacy Mechanisms**: Implement local differential privacy
4. **Real-time Linkage**: Support streaming record linkage scenarios

## Conclusion

FPN-RL represents a significant advancement over traditional threshold-based privacy-preserving record linkage methods. By combining neural network learning capabilities with rigorous differential privacy guarantees, it provides a more flexible, accurate, and privacy-preserving solution for modern record linkage challenges.

The mechanism is particularly valuable for scenarios involving:
- Mixed structured and unstructured data
- Complex similarity relationships that are difficult to capture with predefined metrics
- Need for automatic threshold learning and optimization
- Comprehensive privacy-utility analysis requirements

This implementation provides a foundation for advanced privacy-preserving record linkage research and applications in the cybersecurity domain.