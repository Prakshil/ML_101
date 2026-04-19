# ColumnTransformer Notebook

This folder contains a practical notebook on preprocessing mixed-type tabular data using scikit-learn.

## File
- `columntransformer.ipynb`

## What You Learn
1. Why preprocessing is necessary for machine learning models.
2. How to preprocess data **without** `ColumnTransformer` (manual approach).
3. How to preprocess data **with** `ColumnTransformer` (recommended approach).
4. Why the `ColumnTransformer` workflow is better for reliability and scalability.

## Problem Setup
The dataset includes:
- Numerical columns: `age`, `fever` (with missing values)
- Categorical columns:
  - Ordinal: `cough` (`Mild`, `Strong`)
  - Nominal: `gender`, `city`
- Target column: `has_covid`

## Notebook Structure
### Part A: Without ColumnTransformer
Manual preprocessing pipeline:
- Impute missing values for `fever`
- Ordinal encode `cough`
- One-hot encode `gender` and `city`
- Keep `age` as a passthrough feature
- Concatenate transformed arrays manually

Why this part exists:
- Helps build intuition for each transformation step
- Shows the hidden complexity and repetitive code in manual preprocessing

### Part B: With ColumnTransformer
Unified preprocessing pipeline with a single transformer object:
- Same transformations as Part A
- One central place to define all feature-wise preprocessing

Why this part exists:
- Cleaner code
- Consistent train/test transformation
- Easier maintenance and production readiness

## Key Takeaway
Use manual preprocessing to understand concepts. For real projects, prefer `ColumnTransformer` to reduce errors and keep preprocessing reproducible.
