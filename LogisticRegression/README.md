# Logistic Regression from Scratch using Maximum Likelihood and Cross-Entropy

A complete beginner-friendly implementation of **Logistic Regression from Scratch** using:

- Sigmoid Function
- Maximum Likelihood Estimation (MLE)
- Binary Cross-Entropy Loss (Log Loss)
- Gradient Descent Optimization
- Decision Boundary Visualization
- Training Loss Curve
- Synthetic Dataset using `make_classification`

This project helps in understanding the **mathematics**, **theory**, and **code implementation** behind Logistic Regression without using `sklearn.linear_model.LogisticRegression`.

---

## Table of Contents

- [Introduction](#introduction)
- [What is Logistic Regression?](#what-is-logistic-regression)
- [Why Logistic Regression?](#why-logistic-regression)
- [Linear Model Behind Logistic Regression](#linear-model-behind-logistic-regression)
- [Why Sigmoid Function?](#why-sigmoid-function)
- [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)
- [Binary Cross-Entropy Loss](#binary-cross-entropy-loss)
- [Gradient Descent Optimization](#gradient-descent-optimization)
- [Mathematical Formulas](#mathematical-formulas)
- [Project Workflow](#project-workflow)
- [Code Explanation](#code-explanation)
- [Training Output](#training-output)
- [Decision Boundary](#decision-boundary)
- [Loss Curve](#loss-curve)
- [Advantages](#advantages)
- [Limitations](#limitations)
- [Applications](#applications)
- [Conclusion](#conclusion)

---

## Introduction

Logistic Regression is one of the most important algorithms in Machine Learning used for **binary classification problems**.

It helps answer questions like:
- Will the customer buy or not?
- Is the email spam or not?
- Will the student pass or fail?
- Is the tumor malignant or benign?

Unlike Linear Regression, Logistic Regression predicts **probabilities**, not continuous values. This project implements Logistic Regression completely from scratch using Python and NumPy.

---

## What is Logistic Regression?

Logistic Regression is a **supervised machine learning classification algorithm** used when the output variable has only two classes.

Example:
```text
0 → No
1 → Yes
```
or
```text
0 → Negative
1 → Positive
```

It predicts the probability that a data point belongs to class 1.

---

## Why Logistic Regression?

Suppose we use Linear Regression for classification. It may produce values like:
```text
-3.2
2.7
10.5
```
These are invalid because probability must be between `0` and `1`. That is why we use Logistic Regression. It converts the output into a valid probability using the **Sigmoid Function**.

---

## Linear Model Behind Logistic Regression

The model first calculates a linear equation:

$$ z = w^Tx + b $$

Where:
* $x$ = input features
* $w$ = weights
* $b$ = bias
* $z$ = linear score

This is exactly like Linear Regression. But instead of directly using $z$, we pass it through the Sigmoid function.

---

## Why Sigmoid Function?

The sigmoid function converts any real number into a value between 0 and 1.

Formula:

$$ \sigma(z) = \frac{1}{1+e^{-z}} $$

Output behavior:
* Very negative input → close to 0
* Zero input → 0.5
* Very positive input → close to 1

This makes it perfect for probability prediction.

---

## Maximum Likelihood Estimation (MLE)

The goal of Logistic Regression is:
> Find the best values of weight and bias so that the probability of correct predictions becomes maximum.

This idea is called **Maximum Likelihood Estimation**.

For one sample:

$$ P(y|x) = p^y(1-p)^{1-y} $$

Where:
* $p = P(y=1|x)$
* $y$ is either 0 or 1

For the full dataset:

$$ L(w,b) = \prod_{i=1}^{n} p_i^{y_i}(1-p_i)^{1-y_i} $$

This is called the **Likelihood Function**. We want to maximize this.

---

## Binary Cross-Entropy Loss

Instead of maximizing likelihood directly, we take the log and convert it into a minimization problem. This becomes:

$$ J(w,b) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i\log(p_i) + (1-y_i)\log(1-p_i) \right] $$

This is called:
* Log Loss
* Binary Cross-Entropy Loss
* Negative Log Likelihood

All mean the same thing.

### Why Cross-Entropy?
It heavily penalizes wrong confident predictions.

Example:
If actual label is `y = 1`, then:
- Predicted = 0.99 → very small loss
- Predicted = 0.10 → very large loss

This helps the model learn better.

---

## Gradient Descent Optimization

To minimize the loss, we use Gradient Descent. We calculate:
* derivative with respect to weights
* derivative with respect to bias

Then update them repeatedly.

---

## Mathematical Formulas

### Linear Equation
$$ z = w^Tx + b $$

### Sigmoid Function
$$ \sigma(z) = \frac{1}{1+e^{-z}} $$

### Probability Prediction
$$ P(y=1|x)=\sigma(z) $$

### Cross-Entropy Loss
$$ J(w,b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i\log(p_i) + (1-y_i)\log(1-p_i) \right] $$

### Weight Gradient
$$ \frac{\partial J}{\partial w} = \frac{1}{n} X^T(\hat{p}-y) $$

### Bias Gradient
$$ \frac{\partial J}{\partial b} = \frac{1}{n} \sum(\hat{p}-y) $$

### Update Rule
$$ w := w - \alpha \frac{\partial J}{\partial w} $$
$$ b := b - \alpha \frac{\partial J}{\partial b} $$

Where $\alpha$ is the learning rate.

---

## Project Workflow

The project follows these steps:
- **Step 1:** Generate synthetic binary classification data
- **Step 2:** Split dataset into training and testing sets
- **Step 3:** Initialize weights and bias
- **Step 4:** Apply linear model
- **Step 5:** Apply sigmoid function
- **Step 6:** Compute cross-entropy loss
- **Step 7:** Calculate gradients
- **Step 8:** Update parameters using Gradient Descent
- **Step 9:** Repeat for multiple epochs
- **Step 10:** Predict final classes and evaluate accuracy

---

## Code Explanation

### Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
Used for mathematics, plotting, dataset generation, train-test splitting, and model evaluation.

### Generate Dataset
```python
X, y = make_classification(...)
```
This creates a binary classification dataset with 2 features. Each point belongs to Class 0 or Class 1.

### Initialize Model
```python
self.weights = np.zeros(n_features)
self.bias = 0
```
Initially $w=0$ and $b=0$. The model starts learning from scratch.

### Sigmoid Function
```python
def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
```
Converts score into probability.

### Compute Loss
```python
def compute_loss(self, y, y_pred):
```
Calculates Binary Cross-Entropy Loss. This tells how wrong the model predictions are.

### Gradient Calculation
```python
dw = (1/n) * np.dot(X.T, (y_pred - y))
db = (1/n) * np.sum(y_pred - y)
```
These are derivatives of the loss function. They tell how to update parameters.

### Parameter Update
```python
self.weights -= self.lr * dw
self.bias -= self.lr * db
```
This moves the model toward lower loss.

### Prediction
```python
return np.where(y_proba >= 0.5, 1, 0)
```
If probability is $\ge 0.5 \rightarrow \text{Class 1}$, else $< 0.5 \rightarrow \text{Class 0}$.

---

## Training Output

Example:
```text
Epoch    0 | Loss: 0.693147
Epoch  100 | Loss: 0.412839
Epoch  200 | Loss: 0.331251
...
Final Weights: [ 1.24 -0.87 ]
Final Bias: 0.31
Test Accuracy: 0.90
```
This shows the loss keeps decreasing, the model keeps improving, and the final accuracy is high, meaning the model has converged.

---

## Decision Boundary

The model creates a separating boundary between two classes. The black line represents $P(y=1)=0.5$. This is the classification boundary. One side predicts Class 0, the other side predicts Class 1.

---

## Loss Curve

The training loss graph should move downward. This means the model is learning, the error is reducing, and optimization is working properly. A decreasing curve indicates successful convergence.

---

## Advantages
* Simple and fast
* Easy to interpret
* Works well for binary classification
* Provides probability outputs
* Strong baseline model
* Widely used in real-world ML problems

---

## Limitations
* Works best for linearly separable data
* Struggles with highly complex boundaries
* Sensitive to outliers
* Assumes linear decision boundary
* May underperform on non-linear problems

---

## Applications
Used in:
* Spam Detection
* Fraud Detection
* Medical Diagnosis
* Credit Risk Prediction
* Customer Churn Prediction
* Recommendation Systems
* Sentiment Analysis
* Admission Prediction

---

## Conclusion

Logistic Regression is one of the most fundamental classification algorithms in Machine Learning. It uses linear equations, the sigmoid function, maximum likelihood, cross-entropy loss, and gradient descent to learn the best decision boundary between two classes. Focus on understanding it from scratch gives a strong foundation for Deep Learning and Neural Networks, because the same mathematical intuition is used everywhere.
