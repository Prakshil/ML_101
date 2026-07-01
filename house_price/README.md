# Bengaluru House Price Prediction

A machine learning project for predicting house prices in Bengaluru using data cleaning, feature engineering, outlier handling, and multiple regression models.

## Project Overview

This project uses the dataset in `Bengaluru_House_Data.csv` and a notebook workflow in `house_price.ipynb` to:

- analyze and clean raw housing data
- engineer useful features such as BHK and price per square foot
- remove noisy and unrealistic outliers
- train and compare multiple regression models
- evaluate model quality with standard regression metrics

## Dataset

- File: `Bengaluru_House_Data.csv`
- Key columns commonly used in this notebook workflow:
  - `location`
  - `size`
  - `total_sqft`
  - `bath`
  - `price`

## Workflow Summary

The notebook currently follows this flow:

1. Load data and inspect nulls/value distributions.
2. Drop less-useful columns (`area_type`, `society`, `balcony`, `availability`).
3. Handle missing values:
   - forward fill for `location`
   - default `size` to `2 BHK`
   - median fill for `bath`
4. Convert and engineer features:
   - parse `size` into numeric `bhk`
   - normalize `total_sqft` ranges to numeric values
   - create `price_per_sqft`
5. Group rare locations into `other`.
6. Remove outliers using:
   - sqft-per-bhk threshold
   - location-wise price-per-sqft filtering
   - BHK-based consistency filtering
7. Build preprocessing and modeling pipelines with scikit-learn.
8. Train and evaluate:
   - Linear Regression
   - Lasso Regression
   - Ridge Regression
   - Decision Tree Regressor

## Tech Stack

- Python
- Jupyter Notebook
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## Project Structure

```text
house_price/
|-- Bengaluru_House_Data.csv
|-- house_price.ipynb
|-- requirements.txt
|-- .gitignore
|-- LICENSE
`-- README.md
```

## Setup and Run

### 1) Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the notebook

```bash
jupyter notebook
```

Then open `house_price.ipynb` and run cells in order.

## Model Evaluation

The notebook prints these metrics for each model:

- R2 Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

It also plots Actual vs Predicted values for visual comparison.

## GitHub Upload Checklist

If this folder is not a git repository yet:

```bash
git init
git add .
git commit -m "Initial commit: Bengaluru house price prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Future Improvements

- Add a saved model artifact for direct inference
- Add a small prediction script or API endpoint
- Add cross-validation and hyperparameter tuning
- Track experiments and metrics in a structured way

## License

This project is licensed under the MIT License. See `LICENSE` for details.
