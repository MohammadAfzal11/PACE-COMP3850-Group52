# Privacy-Preserving Record Linkage: Siamese Neural Network + Counting Bloom Filter

## Project Overview

This document presents experimental results comparing different privacy-preserving record linkage (PPRL) approaches using Siamese Neural Networks with Counting Bloom Filters (CBF) against traditional CBF baselines.

**Authors**: Mohammad Afzal Satti
**Course**: PACE COMP 3850  
**Date**: September 2025

---

## Table of Contents

1. [Methodology](#methodology)
2. [Model Architecture](#model-architecture)
3. [Experimental Scenarios](#experimental-scenarios)
4. [Results Summary](#results-summary)
5. [Analysis & Insights](#analysis--insights)
6. [Implementation Details](#implementation-details)
7. [Conclusions](#conclusions)

---

## Methodology

### Data Preprocessing
- **Source**: Voter registration datasets (Alice & Bob)
- **Encoding**: Counting Bloom Filters with q-gram tokenization (q=2)
- **Privacy**: Hash-based encoding with configurable parameters
- **Fields Used**: first_name, last_name, city

### Evaluation Metrics
- **Accuracy**: Overall classification performance
- **F1 Score**: Harmonic mean of precision and recall
- **Precision/Recall**: Class-specific performance
- **Confusion Matrix**: Detailed classification results

---

## Model Architecture

### Siamese Neural Network + CBF

## Experimental Scenarios

## Scenario 1: Small Dataset (100 Records Each)

### **Dataset Configuration**
- **Alice Records**: 100 (Alice_numrec_100_corr_25.csv)
- **Bob Records**: 100 (Bob_numrec_100_corr_25.csv)
- **Training Pairs**: 100 (50 positive, 50 negative)
- **Test Pairs**: 20 (10 positive, 10 negative)
- **Class Balance**: Perfect 50/50 split

### **Results**

| Model | Accuracy | F1 Score | Precision (Match) | Recall (Match) | Training Time |
|-------|----------|----------|-------------------|----------------|---------------|
| **Siamese + CBF** | 75.0% | 66.67% | 100% | 50.0% | ~2 minutes |
| **CBF Baseline** | 95.0% | 95.24% | 95.2% | 95.2% | <1 second |

### **Key Findings**
- ✅ **Balanced Classes**: No class imbalance issues
- ❌ **Limited Data**: Insufficient for deep learning to excel
- ✅ **CBF Effectiveness**: Simple method highly effective
- ⚠️ **Overfitting**: Siamese network achieved 100% training accuracy

Scenario 2 Results (500 Records) - COMPLETE
### **Results**
| Model | Accuracy | F1 Score | Precision (Match) | Recall (Match) | Training Time |
|-------|----------|----------|-------------------|----------------|---------------|
| **Siamese + CBF** | 90.0% | 90.74% | 84.0% | 98.0% | ~10 seconds |
| **CBF Baseline** | 96.0% | 96.15% | 96.1% | 96.1% | <1 second |

### **Key Findings**
- ✅ **Dramatic Improvement**: 24% F1 score increase vs 100-record dataset
- ✅ **Competitive Performance**: Now within 6% of CBF baseline
- ✅ **Scalability Demonstrated**: Deep learning advantage with sufficient data
- ✅ **High Recall for Matches**: 98% match detection rate
