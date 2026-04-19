# Standardization and Normalization

## Overview
This folder contains notebooks on feature scaling techniques used to bring numerical features to a common scale. These preprocessing steps are crucial for many machine learning algorithms.

## Contents
- **Standardization_Normalization.ipynb** - Comprehensive guide to both standardization and normalization techniques
- **std.ipynb** - Deep dive into standardization (Z-score normalization)
- **Housing.csv** - Sample housing dataset for standardization examples
- **wine_data.csv** - Sample wine dataset for normalization examples

## Key Concepts
- **Feature Scaling**: Transforming numerical features to a common scale
- **Standardization**: Transforming features to have mean=0 and std=1
- **Normalization**: Scaling features to a fixed range (typically 0-1)
- **Feature Preprocessing**: Preparation step critical for algorithm performance

## Standardization (Z-score Normalization)
- **Formula**: $z = \frac{x - \mu}{\sigma}$
- **Output Range**: Typically -3 to +3 (unbounded)
- **Mean**: 0
- **Standard Deviation**: 1
- **When to Use**: Most ML algorithms (linear/logistic regression, SVM, KNN)
- **Advantages**: Removes effect of scale, handles outliers reasonably well

## Normalization (Min-Max Scaling)
- **Formula**: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$
- **Output Range**: 0 to 1 (bounded)
- **Advantages**: Bounded range, preserves shape of original distribution
- **Disadvantages**: Sensitive to outliers, new data outside min-max range causes issues
- **When to Use**: Neural networks, image data, algorithms that need bounded ranges

## Other Scaling Techniques
- **Robust Scaler**: Uses median and IQR, robust to outliers
- **Log Scaling**: For right-skewed data
- **Box-Cox Transformation**: Power transformation for normalization
- **Quantile Transformer**: Maps to uniform or normal distribution

## Learning Outcomes
By the end of these notebooks, you will understand:
1. Why feature scaling is important for ML models
2. Differences between standardization and normalization
3. When to apply each technique
4. How to implement scaling in scikit-learn
5. How scaling affects model performance and interpretability

## Prerequisites
- Understanding of mean, variance, and standard deviation
- Familiarity with pandas DataFrames and numpy arrays
- Basic knowledge of ML algorithms
- Understanding of data distributions

## Algorithms Affected by Scale

| Algorithm | Sensitive? | Scaling Needed? |
|-----------|-----------|-----------------|
| Linear Regression | Yes | Recommended |
| Logistic Regression | Yes | Recommended |
| SVM | Yes | Required |
| KNN | Yes | Required |
| Decision Trees | No | No |
| Random Forest | No | No |
| Neural Networks | Yes | Recommended |

## Important Guidelines

1. **Fit on Training Data Only**: Always fit the scaler on training data
2. **Apply to Test Data**: Transform test data using the same scaler
3. **Prevent Data Leakage**: Never fit scaler on entire dataset before splitting
4. **Inverse Transformation**: Use `inverse_transform()` to convert predictions back to original scale
5. **Feature Independence**: Scale each feature independently
6. **Outlier Handling**: Consider handling outliers before scaling

## Implementation Tips
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Common Mistakes to Avoid
- Fitting scaler on entire dataset (causes data leakage)
- Forgetting to scale test data
- Using inappropriate scaler for the algorithm
- Not saving the fitted scaler for production deployment
- Scaling categorical variables (they should be encoded instead)

## Notes
- Feature scaling is essential preprocessing but not a cure-all
- The choice of scaler can impact model performance
- Always visualize data before and after scaling
- Document your scaling choices for reproducibility
- In production, reuse the same scaler that was fitted on training data
