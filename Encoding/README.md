# Encoding

## Overview
This folder contains notebooks on encoding techniques used to transform categorical variables into numerical representations suitable for machine learning models.

## Contents
- **Encoding.ipynb** - Comprehensive guide to encoding methods including:
  - Label Encoding
  - One-Hot Encoding
  - Ordinal Encoding
  - Target Encoding
  - Binary Encoding

## Key Concepts
- **Categorical Data**: Features that represent categories rather than numerical values
- **Encoding Methods**: Different strategies to convert categories into numbers
- **Use Cases**: When to apply each encoding technique based on data characteristics

## Learning Outcomes
By the end of this notebook, you will understand:
1. Why categorical encoding is necessary for ML models
2. Different types of categorical features (nominal vs ordinal)
3. Advantages and limitations of each encoding method
4. How to implement encoding in scikit-learn

## Prerequisites
- Basic understanding of pandas DataFrames
- Familiarity with numpy arrays
- Introduction to machine learning concepts

## Notes
- Always encode categorical variables before training ML models
- The choice of encoding method depends on the feature type and the algorithm used
- Be careful of the curse of dimensionality when using one-hot encoding on high-cardinality features
