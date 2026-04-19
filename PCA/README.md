# Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA)

## Overview
This folder contains notebooks on dimensionality reduction techniques used to reduce the number of features while preserving important information in the data.

## Contents
- **PCA.ipynb** - Principal Component Analysis for unsupervised dimensionality reduction
- **LDA.ipynb** - Linear Discriminant Analysis for supervised dimensionality reduction

## Key Concepts
- **Dimensionality Reduction**: Reducing feature count while retaining information
- **Principal Components**: New features that capture maximum variance in the data
- **Explained Variance**: Proportion of information captured by each component
- **Feature Space Transformation**: Projecting data onto lower-dimensional space

## Principal Component Analysis (PCA)
- **Unsupervised Method**: Doesn't use target variable
- **Variance Maximization**: Finds directions of maximum variance
- **Use Cases**: Data visualization, noise reduction, preprocessing
- **Advantages**: Works with any data type, computationally efficient

## Linear Discriminant Analysis (LDA)
- **Supervised Method**: Uses target variable to find discriminative features
- **Class Separability**: Maximizes between-class variance and minimizes within-class variance
- **Use Cases**: Classification preprocessing, feature extraction for labeled data
- **Advantages**: Often better for classification than PCA when labels are available

## Learning Outcomes
By the end of these notebooks, you will understand:
1. How PCA works and when to use it
2. How LDA differs from PCA and when it's more appropriate
3. How to determine the optimal number of components
4. How to interpret explained variance and loadings
5. Practical implementation using scikit-learn

## Prerequisites
- Understanding of matrix operations and linear algebra
- Familiarity with covariance matrices and eigenvalues
- Basic knowledge of pandas and numpy
- Understanding of variance and standard deviation

## Applications in ML
- **Visualization**: Reducing to 2-3 dimensions for plotting high-dimensional data
- **Noise Reduction**: Removing components with low variance (noise)
- **Computational Efficiency**: Reducing features for faster model training
- **Handling Multicollinearity**: PCA naturally removes correlated features

## Comparison: PCA vs LDA

| Aspect | PCA | LDA |
|--------|-----|-----|
| Supervision | Unsupervised | Supervised |
| Optimization | Maximize variance | Maximize class separability |
| Use Case | Exploration, visualization | Classification preparation |
| Output | Uncorrelated components | Discriminative features |

## Notes
- PCA assumes linear relationships; consider kernel PCA for non-linear data
- Always scale features before applying PCA (affects variance calculations)
- The "explained variance" helps determine how many components to keep
- Components are ordered by variance explained (most to least)
- Interpretation of components can be challenging; examine loadings to understand them
