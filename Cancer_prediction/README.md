# Cancer Prediction with Logistic Regression

## Overview
This project uses the Breast Cancer Wisconsin dataset to build a binary classification model that predicts whether a tumor is benign or malignant.

The workflow is implemented in the notebook and includes:
- Data loading and inspection
- Missing-value analysis
- Data cleaning and target encoding
- Target distribution visualization
- Feature scaling and train-test split
- Logistic Regression training and evaluation

## Project Structure
- `cancer.csv`: Input dataset
- `cancer.ipynb`: End-to-end analysis and model notebook

## Target Variable
The `diagnosis` column is encoded as:
- `0`: Benign
- `1`: Malignant

## Requirements
Install the following Python packages if they are not already available:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## How to Run
1. Open `cancer.ipynb` in VS Code or Jupyter.
2. Run all cells from top to bottom.
3. Review the final model metrics:
   - Confusion Matrix
   - Classification Report
   - Accuracy Score

## Notes
- The notebook uses an idempotent target-encoding step so rerunning cells does not corrupt labels.
- You can improve performance by tuning Logistic Regression hyperparameters or testing other classifiers.
