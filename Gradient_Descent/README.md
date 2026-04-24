# Gradient Descent from Scratch

A complete, beginner-friendly implementation and explanation of **Gradient Descent** in Machine Learning.

This repository demonstrates how Gradient Descent works mathematically and programmatically using Python and NumPy. It includes:

- the basic idea of Gradient Descent
- loss function and optimization concept
- derivatives and update rules
- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-Batch Gradient Descent
- step-by-step code explanation
- mathematical formulas
- working examples
- output interpretation

---

## Table of Contents

- [Introduction](#introduction)
- [What is Gradient Descent?](#what-is-gradient-descent)
- [Why Do We Need Gradient Descent?](#why-do-we-need-gradient-descent)
- [Basic Mathematical Idea](#basic-mathematical-idea)
- [Loss Function](#loss-function)
- [Derivative and Slope](#derivative-and-slope)
- [Parameter Update Rule](#parameter-update-rule)
- [Types of Gradient Descent](#types-of-gradient-descent)
  - [Batch Gradient Descent](#batch-gradient-descent)
  - [Stochastic Gradient Descent](#stochastic-gradient-descent)
  - [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
- [Working of Gradient Descent](#working-of-gradient-descent)
- [Code Explanation](#code-explanation)
- [Example Dataset](#example-dataset)
- [Output Interpretation](#output-interpretation)
- [Advantages](#advantages)
- [Limitations](#limitations)
- [Applications](#applications)
- [Conclusion](#conclusion)

---

## Introduction

Gradient Descent is one of the most important optimization algorithms in Machine Learning and Deep Learning. It is used to minimize the error of a model by updating parameters such as **weight** and **bias**.

In simple words, Gradient Descent helps a model learn the best possible values of its parameters so that predictions become closer to the actual values.

---

## What is Gradient Descent?

Gradient Descent is an iterative optimization technique used to find the **minimum value** of a cost or loss function.

The word can be understood as:

- **Gradient** → slope or direction of steepest change
- **Descent** → moving downward

So Gradient Descent means **moving step by step in the direction that reduces the loss**.

---

## Why Do We Need Gradient Descent?

In machine learning, models make predictions using parameters like weight and bias.

For example, in linear regression:

\[
\hat{y} = wx + b
\]

Here:

- \(x\) = input
- \(w\) = weight
- \(b\) = bias
- \(\hat{y}\) = predicted output

Our goal is to choose the best values of \(w\) and \(b\) so that the predicted value \(\hat{y}\) is as close as possible to the actual value \(y\).

To do that, we need to minimize the loss function. Gradient Descent helps us do exactly that.

---

## Basic Mathematical Idea

The model predicts output using:

\[
\hat{y} = wx + b
\]

The loss tells us how wrong the prediction is.

If the loss is high, the model is performing poorly.
If the loss is low, the model is performing well.

Gradient Descent repeatedly updates \(w\) and \(b\) in such a way that the loss keeps decreasing.

---

## Loss Function

A loss function measures the difference between the actual and predicted values.

For regression problems, one of the most common loss functions is **Mean Squared Error (MSE)**:

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

Where:

- \(n\) = number of samples
- \(y_i\) = actual value
- \(\hat{y}_i\) = predicted value

### Why squared error?

We square the error because:

- it removes negative signs
- it penalizes larger errors more strongly
- it gives a smooth mathematical function for differentiation

---

## Derivative and Slope

Gradient Descent works using derivatives.

A derivative gives the **slope** of a function at a particular point.

For the loss function:

- the derivative with respect to weight gives the slope for weight
- the derivative with respect to bias gives the slope for bias

These slopes tell us whether to increase or decrease the parameters.

### Intuition

- If the slope is positive, the function is going upward.
- If the slope is negative, the function is going downward.

To minimize the loss, we move in the opposite direction of the slope.

---

## Parameter Update Rule

The general update rule in Gradient Descent is:

\[
w := w - \alpha \frac{\partial L}{\partial w}
\]

\[
b := b - \alpha \frac{\partial L}{\partial b}
\]

Where:

- \(w\) = weight
- \(b\) = bias
- \(\alpha\) = learning rate
- \(L\) = loss function

### Meaning

- If the gradient is large, the update step will be larger.
- If the gradient is small, the update step will be smaller.
- Learning rate controls how big each step should be.

---

## Types of Gradient Descent

There are three main types of Gradient Descent:

1. **Batch Gradient Descent**
2. **Stochastic Gradient Descent**
3. **Mini-Batch Gradient Descent**

Each type differs in how much data it uses for one update.

---

## Batch Gradient Descent

### Definition

Batch Gradient Descent uses the **entire training dataset** to compute the gradient for one parameter update.

### Working

For each epoch:

1. take all data points
2. compute predictions
3. calculate total loss
4. compute gradients using all samples
5. update weight and bias

### Mathematical Form

\[
\frac{\partial L}{\partial w} = \frac{-2}{n}\sum x_i(y_i - \hat{y}_i)
\]

\[
\frac{\partial L}{\partial b} = \frac{-2}{n}\sum (y_i - \hat{y}_i)
\]

### Characteristics

- stable updates
- slower for large datasets
- uses more memory
- good for small datasets

---

## Stochastic Gradient Descent

### Definition

Stochastic Gradient Descent updates parameters using **one training example at a time**.

### Working

For each sample:

1. take one input-output pair
2. compute prediction
3. calculate loss
4. update parameters immediately

### Mathematical Form

For one sample:

\[
\frac{\partial L}{\partial w} = -2x(y - \hat{y})
\]

\[
\frac{\partial L}{\partial b} = -2(y - \hat{y})
\]

### Characteristics

- very fast
- noisy updates
- can jump around the minimum
- useful for very large datasets

---

## Mini-Batch Gradient Descent

### Definition

Mini-Batch Gradient Descent uses a **small group of samples** at a time.

### Working

Instead of using the full dataset or just one sample:

- the dataset is divided into small batches
- each batch is used for one update
- the model is updated after every batch

### Mathematical Form

For a batch of size \(m\):

\[
\frac{\partial L}{\partial w} = \frac{-2}{m}\sum x_i(y_i - \hat{y}_i)
\]

\[
\frac{\partial L}{\partial b} = \frac{-2}{m}\sum (y_i - \hat{y}_i)
\]

### Characteristics

- faster than batch GD
- more stable than SGD
- widely used in deep learning
- efficient and practical

---

## Working of Gradient Descent

The full working process is as follows:

### Step 1: Initialize Parameters
Start with random values or zero values for weight and bias.

\[
w = 0,\quad b = 0
\]

### Step 2: Make Prediction
Use the linear model:

\[
\hat{y} = wx + b
\]

### Step 3: Calculate Loss
Find the error between actual and predicted values using MSE.

### Step 4: Compute Gradient
Find the slope of the loss function with respect to weight and bias.

### Step 5: Update Parameters
Move parameters in the direction that reduces loss.

### Step 6: Repeat
Repeat the process until the loss becomes minimum or the maximum number of epochs is reached.

---

## Code Explanation

Here is a step-by-step breakdown of the Python implementations for each variant of Gradient Descent from the provided Jupyter Notebooks.

### 1. Batch Gradient Descent (`batchGD.ipynb`)

In Batch Gradient Descent, the entire dataset is used in each epoch to calculate the gradients and update the parameters.

**Formula applied in code:**
The gradients over all $n$ samples:
$$dw = \frac{-2}{n} \sum (x_i \cdot (y_i - \hat{y}_i))$$
$$db = \frac{-2}{n} \sum (y_i - \hat{y}_i)$$

**Code Breakdown:**
```python
import numpy as np

# 1. Dataset Initialization
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([5, 7, 9, 11, 13], dtype=float)

# 2. Parameter Initialization
w = 0
b = 0

# 3. Hyperparameters
learning_rate = 0.01
epochs = 1000
n = len(X)

# 4. Training Loop
for epoch in range(epochs):
    # Make Prediction for the entire batch
    y_pred = w * X + b

    # Calculate Gradients using the full dataset
    dw = (-2/n) * np.sum(X * (Y - y_pred))
    db = (-2/n) * np.sum(Y - y_pred)

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Final Weight:", w)
print("Final Bias:", b)
```
**Explanation:** 
- `y_pred = w * X + b`: Calculates the predictions for all inputs simultaneously using array broadcasting.
- `dw` and `db`: Calculate the mean error gradients using `np.sum()` across the entire batch $X$ and $Y$.
- Parameters are updated once per epoch. This yields a steady error reduction but can be memory-heavy on large datasets.

---

### 2. Stochastic Gradient Descent (`Stochastic.ipynb`)

In Stochastic Gradient Descent (SGD), the parameters are updated for every single data point in the dataset.

**Formula applied in code:**
For a single sample $(x_i, y_i)$:
$$dw = -2 \cdot x_i \cdot (y_i - \hat{y}_i)$$
$$db = -2 \cdot (y_i - \hat{y}_i)$$

**Code Breakdown:**
```python
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([5, 7, 9, 11, 13], dtype=float)

w = 0
b = 0
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # Iterate through every single data point
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        # Prediction for one sample
        y_pred = w * x + b

        # Gradients for one sample
        dw = -2 * x * (y - y_pred)
        db = -2 * (y - y_pred)

        # Immediate parameter update
        w = w - learning_rate * dw
        b = b - learning_rate * db
```
**Explanation:** 
- Instead of using `np.sum()`, SGD uses a nested `for` loop to look at each point `X[i]` and `Y[i]` individually.
- `dw` and `db` are computed for just that one point without division by $n$.
- Weights `w` and `b` update immediately. This creates much faster but highly erratic updates.

---

### 3. Mini-Batch Gradient Descent (`minibatchGD.ipynb`)

Mini-Batch Gradient Descent balances the extremes by processing small subsets of data (batches) at a time.

**Formula applied in code:**
For a batch of size $m$:
$$dw = \frac{-2}{m} \sum_{i=1}^{m} x_i \cdot (y_i - \hat{y}_i)$$
$$db = \frac{-2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)$$

**Code Breakdown:**
```python
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([5, 7, 9, 11, 13], dtype=float)

w = 0
b = 0
learning_rate = 0.01
epochs = 100
batch_size = 2 # Size of the mini-batch

for epoch in range(epochs):
    # Step through dataset in chunks of 'batch_size'
    for i in range(0, len(X), batch_size):
        # Extract the mini-batch
        X_batch = X[i : i+batch_size]
        Y_batch = Y[i : i+batch_size]

        # Prediction for the mini-batch
        y_pred = w * X_batch + b
        
        n = len(X_batch)

        # Gradients using mini-batch data
        dw = (-2/n) * np.sum(X_batch * (Y_batch - y_pred))
        db = (-2/n) * np.sum(Y_batch - y_pred)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
```
**Explanation:** 
- The second `for` loop iterates using a step equal to `batch_size`.
- `X_batch` and `Y_batch` slice the main arrays. 
- The gradients and updates are processed just like in Batch GD but applied only to the smaller subset ($n$ is now the batch length, protecting against errors on non-divisible final batches).

---

## Example Dataset

In the code files, we use a simple set of \(x\) and \(y\) values:
- `X` = [1, 2, 3, 4, 5]
- `Y` = [5, 7, 9, 11, 13]

This follows a clear linear relationship: **`Y = 2X + 3`** (meaning the ideal weight \(w\) is 2 and the ideal bias \(b\) is 3). All versions of our code successfully converge close to these ideal values.

---

## Output Interpretation

Across all implementations, printing the weights should approach:
- **Final Weight**: ~2.0
- **Final Bias**: ~3.0

The speed at which it converges depends on the variant (SGD reaches it faster but with more noise, whereas Mini-batch provides a balanced speed/stability ratio).

---

## Key Formula Summary

Here is the central reference for Gradient Descent:

1. **Prediction Rule (Linear Model):**  
   $$\hat{y} = wx + b$$

2. **Mean Squared Error (Loss function):**  
   $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

3. **Gradient Descent Updates:**  
   $$w_{new} = w - \alpha \cdot \frac{\partial L}{\partial w}$$  
   $$b_{new} = b - \alpha \cdot \frac{\partial L}{\partial b}$$

4. **Derivatives:**  
   - $\frac{\partial L}{\partial w} = \frac{-2}{n}\sum x \cdot (y - \hat{y})$ (For batch of size $n$, $n=1$ for SGD)
   - $\frac{\partial L}{\partial b} = \frac{-2}{n}\sum (y - \hat{y})$

---

## Advantages
- Memory efficient for very large datasets (especially SGD and Mini-batch).
- Simple concept easily extended to advanced optimization (Adam, RMSProp).
- Extremely scalable.

## Limitations
- Can get stuck in "local minima" (though mostly an issue for non-convex functions like deep neural nets, not linear regression).
- Highly sensitive to the learning rate $\alpha$. Too large = explodes (diverges). Too small = converges too slowly.
- Requires feature scaling (normalization/standardization) for optimal performance.

## Applications
- Linear Regression and Logistic Regression models.
- Core mechanism training Deep Neural Networks (Backpropagation).
- Support Vector Machines (SVMs) and Neural network embedding updates.

---

## Conclusion

Gradient Descent is the foundational learning algorithm behind almost all of modern machine learning. By simply breaking the algorithm into computing the error, taking its derivative (finding exactly which way is 'down' the error hill), and iteratively updating weights, complex datasets can be modeled gracefully. 

Understanding the trade-offs between **Batch** (stable, slow), **Stochastic** (noisy, fast, memory-light), and **Mini-Batch** (the practical middle-ground) gives you complete control over training neural networks and classic regression models efficiently.