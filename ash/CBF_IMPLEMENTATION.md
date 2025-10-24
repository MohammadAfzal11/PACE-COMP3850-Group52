# Content-Based Filtering (CBF) Implementation for Medical Record Linkage

## Overview
This implementation enhances the medical record linkage system by incorporating Content-Based Filtering (CBF) to extract and compare comprehensive medical information from unstructured text, moving beyond simple demographic matching (age/gender).

## Key Changes

### 1. Enhanced Medical Information Extraction
**Previous approach:**
- Only extracted 2 demographics: age and gender
- Used simple regex pattern: `[[X, 'year']] M`

**New approach:**
- Extracts **7 categories** of medical information:
  1. **Demographics**: Age and gender (with 4 different pattern variations)
  2. **Diagnoses**: 50+ medical conditions (diabetes, cancer, heart disease, etc.)
  3. **Symptoms**: 40+ clinical presentations (pain, fever, dyspnea, etc.)
  4. **Procedures**: 30+ medical interventions (surgery, MRI, chemotherapy, etc.)
  5. **Medications**: 30+ drugs and drug classes (insulin, aspirin, clozapine, etc.)
  6. **Test Results**: Lab values (hemoglobin, creatinine, blood pressure, etc.)
  7. **Body Parts**: 35+ anatomical locations (heart, kidney, liver, etc.)

### 2. Content-Based Filtering Features
Implemented 6 new CBF similarity features:
- `cbf_diagnoses_sim`: Jaccard similarity of diagnoses
- `cbf_symptoms_sim`: Jaccard similarity of symptoms
- `cbf_procedures_sim`: Jaccard similarity of procedures
- `cbf_medications_sim`: Jaccard similarity of medications
- `cbf_body_parts_sim`: Jaccard similarity of body parts
- `cbf_overall_similarity`: Weighted average (weights: 0.3, 0.2, 0.2, 0.15, 0.15)

### 3. Updated Feature Set
**Total features increased from 10 to 16:**

Original features (10):
- tfidf_cosine
- hashed_jaccard
- length_diff_pct
- age_diff
- gender_match_same
- gender_match_diff
- gender_match_unknown

New CBF features (6):
- cbf_diagnoses_sim
- cbf_symptoms_sim
- cbf_procedures_sim
- cbf_medications_sim
- cbf_body_parts_sim
- cbf_overall_similarity

## Technical Implementation

### Function: `extract_medical_information(text)`
Parses unstructured medical text and returns a dictionary with all extracted features.

**Example:**
```python
text = "A 45-year-old male with diabetes and hypertension underwent cardiac catheterization."
info = extract_medical_information(text)
# Returns:
# {
#   'age': 45.0,
#   'gender': 'M',
#   'diagnoses': ['diabetes', 'hypertension'],
#   'symptoms': [],
#   'procedures': ['catheterization'],
#   'medications': [],
#   'body_parts': ['cardiac']
# }
```

### Function: `calculate_medical_feature_similarity(list1, list2)`
Calculates Jaccard similarity between two lists of medical features.

**Formula:** 
```
similarity = |intersection(A, B)| / |union(A, B)|
```

**Example:**
```python
diagnoses1 = ['diabetes', 'hypertension']
diagnoses2 = ['diabetes', 'heart disease']
sim = calculate_medical_feature_similarity(diagnoses1, diagnoses2)
# Returns: 0.3333 (1 common / 3 total)
```

## Validation Results

### Test Case 1: Dissimilar Records
- Record 1: 57-year-old male with thyroid issues, fatigue, dyspnoea
- Record 2: 22-year-old male with renal calculus, surgical history
- **CBF Overall Similarity: 0.05** (very low - correctly identified as non-matching)

### Test Case 2: Similar Records
- Record 1: 45-year-old male with diabetes, hypertension, chest pain, cardiac catheterization
- Record 2: 45-year-old male with diabetes, hypertension, chest pain, angioplasty
- **CBF Overall Similarity: 0.64** (high - correctly identified as matching)

### Similarity Breakdown (Test Case 2):
- Diagnoses: 1.00 (perfect match)
- Symptoms: 0.60 (good overlap)
- Procedures: 0.33 (different but related)
- Medications: 1.00 (perfect match)

## Benefits of CBF Approach

1. **More Accurate Matching**: Uses medical context beyond demographics
2. **Robust to Missing Data**: If demographics are missing, CBF features can still identify matches
3. **Interpretable**: Each similarity score has clear medical meaning
4. **Feature Importance**: Random Forest can identify which medical features are most predictive
5. **Privacy-Preserving**: Works with anonymized text and differential privacy

## Files Modified

1. **ash/working code with random forest and DP.ipynb**
   - Updated cells 0, 3, 6, 10, 15, 17, 18

2. **ash/working_code_with_random_forest_and_dp.py**
   - Updated all corresponding functions

3. **README.md**
   - Added documentation for ash folder and CBF features

## Compatibility with Existing Features

The implementation maintains all existing functionality:
- ✅ TF-IDF vectorization still used for text similarity
- ✅ Hashed n-grams for privacy-preserving similarity
- ✅ Differential privacy with Laplace noise
- ✅ Random Forest classifier
- ✅ Multiple model comparisons (threshold-based, improved threshold, Random Forest)
- ✅ Privacy-utility tradeoff analysis (multiple epsilon values)
- ✅ Feature importance analysis
- ✅ ROC curves and confusion matrices

## Usage

### Running the Notebook
```bash
cd ash/
jupyter notebook "working code with random forest and DP.ipynb"
```

### Key Cells to Run in Order
1. Cell 0: Load data and extract comprehensive medical information
2. Cell 1-2: Text preprocessing and hashed n-grams
3. Cell 3: Calculate CBF similarities
4. Cell 4: TF-IDF vectorization
5. Cell 5: Calculate age/gender features
6. Cell 6: Prepare features and train Random Forest
7. Rest: Evaluation, visualization, differential privacy analysis

## Future Enhancements

Potential improvements:
1. Add more medical keywords (e.g., specific drug names, rare conditions)
2. Use medical ontologies (SNOMED-CT, ICD codes) for better matching
3. Implement semantic similarity using medical embeddings (BioBERT)
4. Add temporal features (date of diagnosis, duration of symptoms)
5. Include laboratory value comparison (not just detection)
6. Weight adjustments based on dataset characteristics

## References

- Content-Based Filtering: Uses feature overlap to measure similarity
- Jaccard Similarity: Standard set similarity metric
- Medical Text Mining: Extraction of structured information from clinical notes
- Privacy-Preserving Record Linkage: Matching records while protecting sensitive data
