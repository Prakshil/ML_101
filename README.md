# ML_101 🧠

A comprehensive, hands-on machine learning repository covering everything from **mathematical foundations** and **data preprocessing** to **core algorithms** implemented from scratch and **end-to-end mini projects**. Every topic includes beginner-friendly Jupyter notebooks with detailed explanations, code, and visualizations.

---

## 📂 Repository Structure

```
ML_101/
├── Numpy/                          # NumPy fundamentals & advanced operations
├── Pandas/                         # Pandas Series, GroupBy, Merging
├── Maths/                          # Probability, CLT, Confidence Intervals
├── Data_Cleaning-Visualization/    # EDA, data cleaning, interpretation
├── Encoding/                       # Categorical encoding techniques
├── Standardization-Normalization/  # Feature scaling methods
├── Outlier_removal/                # Outlier detection & removal
├── ColumnTransformer/              # ColumnTransformer & pipelines
├── LinearRegression/               # Simple, Multiple, Polynomial LR
├── Gradient_Descent/               # Batch, Stochastic, Mini-batch GD
├── LogisticRegression/             # Logistic Regression & Perceptron
├── Regularization_algos/           # Ridge, Lasso, ElasticNet
├── PCA/                            # PCA & LDA dimensionality reduction
├── Cancer_prediction/              # Mini project: Breast Cancer classification
├── house_price/                    # Mini project: Bengaluru House Price prediction
├── CarPrice_Prediction.ipynb       # Mini project: Used car price prediction
├── cricket_score.ipynb             # Mini project: Cricket score prediction
└── EDA_Handbook.docx               # EDA reference guide
```

---

## 🗺️ Recommended Learning Path

The repository is organized as a **structured curriculum**. Follow the order below for the best learning experience:

### 1️⃣ Foundations — Tools & Math

| Topic | Notebook(s) | Key Concepts |
|-------|------------|--------------|
| **NumPy** | `numpy_fundamentals.ipynb`, `numpy_advanced.ipynb` | Array creation, indexing, slicing, broadcasting, vectorization, memory efficiency |
| **Pandas** | `pandas_series.ipynb`, `groupby.ipynb`, `merging.ipynb` | Series, DataFrames, GroupBy aggregations, merge/join/concat |
| **Mathematics** | `probablity_functions.ipynb`, `central_limit_thm.ipynb`, `confidence_intervals.ipynb` | Probability distributions, Central Limit Theorem, confidence intervals, margin of error |

**Code snippet — NumPy vectorization speedup:**
```python
import numpy as np
import time

size = 10_000_000
arr = np.random.randn(size)

# Python loop — slow
start = time.time()
result_py = sum(arr) / len(arr)
print(f"Python loop: {time.time() - start:.3f}s")

# NumPy vectorized — fast
start = time.time()
result_np = np.mean(arr)
print(f"NumPy: {time.time() - start:.3f}s")
```

**Code snippet — Central Limit Theorem demonstration:**
```python
import numpy as np
import matplotlib.pyplot as plt

population = np.random.exponential(scale=2, size=1_000_000)

sample_means = []
for _ in range(10_000):
    sample = np.random.choice(population, size=30)
    sample_means.append(np.mean(sample))

plt.hist(sample_means, bins=50, density=True)
plt.title("CLT: Sample Means Converge to Normal")
plt.show()
```

---

### 2️⃣ Data Preprocessing & Cleaning

| Topic | Notebook(s) | Key Concepts |
|-------|------------|--------------|
| **Data Cleaning & EDA** | `data_accessing_and_cleaning.ipynb`, `EDA_Basics.ipynb`, `Data_interpretation.ipynb`, `EDA_Practice.ipynb` | Missing values, duplicates, data types, feature engineering, Titanic dataset |
| **Encoding** | `Encoding.ipynb` | Label Encoding, One-Hot Encoding, Ordinal Encoding, Target Encoding, Binary Encoding |
| **Standardization / Normalization** | `Standardization_Normalization.ipynb`, `std.ipynb` | Min-Max scaling, Z-score standardization, Robust Scaler, Absolute Maximum Scaling |
| **Outlier Removal** | `outliers_removal.ipynb`, `zscore_outlier.ipynb` | IQR method, Z-score method, percentile clipping |
| **ColumnTransformer** | `columntransformer.ipynb` | ColumnTransformer, pipelines, mixed data types |

**Code snippet — Standardization (Z-score) vs Normalization (Min-Max):**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

data = pd.DataFrame({'age': [25, 30, 45, 60, 35], 'salary': [30000, 50000, 80000, 120000, 60000]})

# Standardization (Z-score)
scaler_std = StandardScaler()
data_standardized = scaler_std.fit_transform(data)

# Normalization (Min-Max)
scaler_mm = MinMaxScaler()
data_normalized = scaler_mm.fit_transform(data)
```

**Code snippet — Outlier detection using IQR:**
```python
def find_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] < lower) | (data[column] > upper)]

outliers = find_outliers_iqr(df, 'cgpa')
print(f"Outliers found: {len(outliers)}")
```

**Code snippet — ColumnTransformer pipeline:**
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

numeric_features = ['age', 'fever']
categorical_features = ['cough', 'gender']
ordinal_features = ['city']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features),
    ('ord', OrdinalEncoder(), ordinal_features)
])

pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression())])
```

---

### 3️⃣ Supervised Learning — Regression

| Topic | Notebook(s) | Key Concepts |
|-------|------------|--------------|
| **Simple Linear Regression** | `simpleLR.ipynb` | Normal equation, OLS, `fit()`/`predict()` from scratch |
| **Multiple Linear Regression** | `MultipleLR.ipynb` | Matrix form, closed-form solution, R² score |
| **Polynomial Regression** | `PolynomialLR.ipynb` | PolynomialFeatures, non-linear curve fitting |

**Code snippet — Linear Regression from scratch using Normal Equation:**
```python
import numpy as np

class LinearRegressionScratch:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ np.r_[self.intercept_, self.coef_]

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
```

---

### 4️⃣ Optimization — Gradient Descent

| Topic | Notebook(s) | Key Concepts |
|-------|------------|--------------|
| **Batch Gradient Descent** | `batchGD.ipynb` | Full-batch update, cost history, convergence |
| **Stochastic Gradient Descent** | `Stochastic.ipynb` | Per-sample update, noisy convergence |
| **Mini-Batch Gradient Descent** | `minibatchGD.ipynb` | Batch-size tradeoff, data shuffling |

**Code snippet — Batch Gradient Descent:**
```python
def batch_gradient_descent(X, y, lr=0.01, epochs=1000):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(X_b.shape[1])
    cost_history = []

    for epoch in range(epochs):
        gradients = (2/m) * X_b.T @ (X_b @ theta - y)
        theta -= lr * gradients
        cost = np.mean((X_b @ theta - y) ** 2)
        cost_history.append(cost)

    return theta, cost_history
```

---

### 5️⃣ Supervised Learning — Classification

| Topic | Notebook(s) | Key Concepts |
|-------|------------|--------------|
| **Logistic Regression** | `logisticreg.ipynb` | Sigmoid, MLE, cross-entropy loss, decision boundary |
| **Perceptron** | `perceptron_logreg.ipynb` | Step function, Perceptron trick, stochastic updates |
| **Perceptron + Sigmoid** | `perceptron-logreg-sigmoid.ipynb` | Smooth activation, probability outputs |

**Code snippet — Logistic Regression from scratch:**
```python
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear = X @ self.weights + self.bias
            y_pred = self._sigmoid(linear)
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = X @ self.weights + self.bias
        return (self._sigmoid(linear) >= 0.5).astype(int)
```

---

### 6️⃣ Regularization

| Topic | Notebook(s) | Key Concepts |
|-------|------------|--------------|
| **Ridge (L2)** | `regularization.ipynb` | L2 penalty, coefficient shrinkage |
| **Lasso (L1)** | `regularization.ipynb` | L1 penalty, feature selection, sparsity |
| **ElasticNet** | `regularization.ipynb` | L1 + L2 combination, hyperparameter α/λ |

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.01)
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)

for model in [ridge, lasso, elastic]:
    model.fit(X_train, y_train)
    print(f"{model.__class__.__name__}: R² = {model.score(X_test, y_test):.3f}")
```

---

### 7️⃣ Dimensionality Reduction

| Topic | Notebook(s) | Key Concepts |
|-------|------------|--------------|
| **PCA** | `PCA.ipynb` | Unsupervised, variance preservation, eigenvalue decomposition |
| **LDA** | `LDA.ipynb` | Supervised, class separability, discriminant directions |

```python
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_train)
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_train, y_train)
```

---

### 8️⃣ Mini Projects

| Project | File | Description |
|---------|------|-------------|
| **Breast Cancer Prediction** | `Cancer_prediction/cancer.ipynb` | Binary classification (benign vs malignant) using Logistic Regression on the Wisconsin dataset (30 features, 569 samples). Achieves ~97% accuracy. |
| **Bengaluru House Price** | `house_price/house_price.ipynb` | End-to-end regression project: data cleaning, feature engineering (BHK, price_per_sqft), outlier removal, model comparison (Linear, Ridge, Lasso, Decision Tree). |
| **Car Price Prediction** | `CarPrice_Prediction.ipynb` | Used car price prediction with data cleaning and regression on `quikr_car.csv`. |
| **Cricket Score Prediction** | `cricket_score.ipynb` | Parsing YAML-like innings data for cricket match analysis and score prediction. |

**Code snippet — Breast Cancer classification with PCA + Logistic Regression:**
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
print(classification_report(y_test, model.predict(X_test_pca)))
```

**Code snippet — House price outlier removal by location:**
```python
def remove_price_per_sqft_outliers(df):
    output = pd.DataFrame()
    for location, subset in df.groupby('location'):
        mean = subset['price_per_sqft'].mean()
        std = subset['price_per_sqft'].std()
        filtered = subset[(subset['price_per_sqft'] > mean - std) &
                          (subset['price_per_sqft'] < mean + std)]
        output = pd.concat([output, filtered])
    return output

df_clean = remove_price_per_sqft_outliers(df)
```

---

## 🛠️ Tech Stack

| Library | Usage |
|---------|-------|
| **Python 3** | Core language |
| **NumPy** | Numerical computing, linear algebra, from-scratch implementations |
| **Pandas** | Data manipulation, CSV handling, GroupBy, merging |
| **Matplotlib / Seaborn** | Data visualization, loss curves, decision boundaries |
| **Scikit-learn** | Preprocessing, pipelines, model comparison, metrics |
| **Jupyter Notebook** | Interactive learning environment |
| **category_encoders** | Additional encoding techniques (Target, Binary) |
| **SciPy** | Statistical functions, confidence intervals |

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Prakshil/ML_101.git
cd ML_101

# Create a virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Create a virtual environment (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas>=2.0 numpy>=1.24 seaborn>=0.12 matplotlib>=3.7 scikit-learn>=1.3 jupyter>=1.0 ipykernel>=6.0

# Launch Jupyter
jupyter notebook
```

Then open any `.ipynb` file and run cells sequentially.

---

## 🧪 Running Specific Topics

Each folder is **self-contained**. Navigate to the topic of interest and open its notebook:

```bash
# Example: Learn about Gradient Descent
cd Gradient_Descent
jupyter notebook batchGD.ipynb
```

---

## 📚 What Makes This Repo Unique

- **From-scratch implementations** — Linear Regression, Logistic Regression, and all 3 variants of Gradient Descent are implemented using only NumPy, so you understand the math behind the algorithms
- **Code + Theory** — Every topic has detailed markdown explanations alongside code
- **Real datasets** — Breast Cancer Wisconsin, Bengaluru Housing, Titanic, Wine, and more
- **Visual learning** — Loss curves, decision boundaries, coefficient paths, confusion matrices
- **Progressive complexity** — Start with NumPy arrays, end with production-ready ColumnTransformer pipelines

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](./house_price/LICENSE) file for details.
