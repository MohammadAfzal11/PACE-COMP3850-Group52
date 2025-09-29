Privacy-Preserving Record Linkage: Differential Privacy Counting Bloom Filter
Project Overview
This document presents experimental results for privacy-preserving record linkage (PPRL) using Differential Privacy with Counting Bloom Filters, providing mathematical privacy guarantees while maintaining high utility performance.

Authors: Mohammad Afzal Satti
Course: PACE COMP 3850
Date: September 2025

Table of Contents
Methodology

Model Architecture

Experimental Scenarios

Results Summary

Analysis & Insights

Implementation Details

Conclusions

Methodology
Data Preprocessing
Source: Voter registration datasets (Alice & Bob)

Encoding: Counting Bloom Filters with q-gram tokenization (q=2)

Privacy: Differential privacy with Laplace noise mechanism

Fields Used: first_name, last_name, city

Evaluation Metrics
Accuracy: Overall classification performance

F1 Score: Harmonic mean of precision and recall

Privacy Parameter (ε): Controls privacy-utility trade-off

Privacy Loss: Quantified utility degradation for privacy protection

Model Architecture
Differential Privacy Counting Bloom Filter
python
# Core Architecture
Input: Raw Record → CBF Encoding → Laplace Noise Addition → Private CBF Vector

Privacy Mechanism:
├── Sensitivity Calculation: Δf = 0.15 (global sensitivity)
├── Noise Generation: Laplace(0, Δf/ε)
├── Noise Addition: Private_CBF = Clean_CBF + Noise
└── Non-negativity Constraint: max(0, Private_CBF)

Similarity Computation:
├── Dice Coefficient: 2×|CBF1∩CBF2| / (|CBF1|+|CBF2|)
└── Threshold Classification: match = similarity > threshold

Total Parameters: Hash-based encoding (no trainable parameters)
Privacy Guarantee: ε-differential privacy
Experimental Scenarios
Scenario 1: Small Dataset (100 Records Each)
Dataset Configuration
Alice Records: 100 (Alice_numrec_100_corr_25.csv)

Bob Records: 100 (Bob_numrec_100_corr_25.csv)

Test Pairs: 79 (40 positive, 39 negative)

Class Balance: Balanced 50/50 split

Privacy Levels: 5 different ε values (0.5 to 10.0)

Results
Privacy Level	Epsilon (ε)	Accuracy	F1 Score	Privacy Cost	Training Time
Very High	      0.5	    94.94%	     94.87%	      2.63% loss	~1 second
High	          1.0	    97.47%	     97.50%	      0.00% loss	~1 second
Medium	          2.0	    97.47%	     97.56%	     -0.06% loss	~1 second
Low	              5.0	    97.47%	     97.50%	      0.00% loss	~1 second
Very Low	      10.0	    97.47%	     97.50%	      0.00% loss	~1 second

Key Findings
 Excellent Privacy-Utility Balance: Minimal utility loss even at high privacy

 Optimal Performance: 97.56% F1 score with medium privacy (ε=2.0)

 Fast Inference: Sub-second processing time

 Mathematical Guarantees: Formal ε-differential privacy protection

Scenario 2: Medium Dataset (500 Records Each)
Dataset Configuration
Alice Records: 500 (Alice_numrec_500_corr_25.csv)

Bob Records: 500 (Bob_numrec_500_corr_25.csv)

Test Pairs: 200 (100 positive, 100 negative)

Class Balance: Perfect 50/50 split

Privacy Levels: 5 different ε values (0.5 to 10.0)

Results
Privacy Level	Epsilon (ε)	Accuracy	F1 Score	Privacy Cost	Training Time
Very High	      0.5	     94.00%	     93.94%	     2.61% loss	     ~2 seconds
High	          1.0	     95.00%	     95.00%	     1.55% loss	     ~2 seconds
Medium	          2.0	     95.50%	     95.52%	     1.03% loss	     ~2 seconds
Low	              5.0	     96.00%	     96.04%	     0.51% loss	     ~2 seconds
Very Low	      10.0	     96.50%	     96.55%	     0.00% loss	     ~2 seconds

Key Findings
 Consistent Performance: Reliable across different dataset sizes
 Scalable Privacy: Maintained strong privacy protection at scale

Predictable Trade-offs: Clear privacy-utility relationship

Production Ready: Suitable for real-world deployment

Results Summary
Dataset Size Comparison Analysis
Comprehensive comparison showing counter-intuitive finding that 100-record dataset achieves superior performance across all privacy levels.

Cross-Dataset Performance Comparison
Dataset Size	Best Configuration	  Accuracy	  F1 Score	Key Advantage
100 Records	ε=2.0 (Medium Privacy)	  97.47%	   97.56%	+2.04% F1 vs 500 records
500 Records	ε=10.0 (Very Low Privacy) 96.50%	   96.55%	 Better scalability
Privacy-Utility Trade-off Analysis
100 Records Dataset:

Maximum Privacy Cost: 2.63% F1 loss at ε=0.5 (very high privacy)

Optimal Balance: ε=1.0-2.0 providing 97.5%+ performance

Zero Privacy Cost: Achieved at ε≥1.0

500 Records Dataset:

Maximum Privacy Cost: 2.61% F1 loss at ε=0.5 (very high privacy)

Optimal Balance: ε=2.0-5.0 providing 95-96% performance

Gradual Improvement: Performance increases with decreasing privacy

Analysis & Insights
Counter-Intuitive Discovery: Size vs Performance
Key Finding: Smaller datasets achieved superior privacy-utility trade-offs across all privacy levels.

Dataset Size Impact
100 vs 500 Records at ε=1.0: +2.50% F1 advantage for smaller dataset

Consistent Pattern: 100-record dataset outperformed across all ε values

Maximum Gap: 2.5% performance difference at high privacy settings

Theoretical Explanation
Noise-to-Signal Ratio: Lower noise accumulation in smaller CBF vectors

Parameter Optimization: CBF parameters well-suited for 100-record scale

Data Quality: Fewer edge cases and outliers in smaller datasets

Calibration Effect: Differential privacy noise calibration optimal at smaller scale

Technical Achievements
Superior Performance: 97.56% F1 score with formal privacy guarantees

Mathematical Rigor: ε-differential privacy with provable bounds

Practical Efficiency: Sub-second inference for real-time applications

Scalable Design: Effective across 100-500 record datasets