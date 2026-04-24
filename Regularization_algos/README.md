# Regularization in Linear Regression

Regularization adds a penalty to the loss function to **prevent overfitting**, especially when dealing with many or correlated features. The most common forms are **Ridge (L2)**, **Lasso (L1)**, and **Elastic Net (L1+L2)**. These methods shrink large coefficients (and in Lasso’s case, set some to zero) to improve generalization【23†L233-L241】【17†L239-L247】. In summary, regularization helps models avoid fitting noise and address multicollinearity by adding constraints on the weight vector.

## Intuitive Concept

- **Why regularize?** In ordinary least squares (OLS), the model can overfit noisy data or unstable when features are correlated【23†L233-L241】【10†L548-L554】. Regularization adds a penalty for large weights, effectively reducing variance at the cost of a bit more bias (bias–variance tradeoff【17†L278-L287】【23†L333-L342】).
- **Shrinkage:** Ridge (L2) shrinks coefficients smoothly toward zero, while Lasso (L1) can shrink some coefficients exactly to zero, performing implicit feature selection【23†L344-L353】【17†L239-L247】.
- **Elastic Net:** Combines both L1 and L2 penalties to gain advantages of both – i.e. sparsity of Lasso and stability of Ridge【6†L603-L611】【23†L365-L373】.
- **Analogy:** Imagine fitting a flexible curve to data: unconstrained fitting (OLS) can wiggle too much (overfit). Regularization adds a tether, pulling the curve (coefficients) taut and preventing wild swings.

## Formal Definitions

Given a dataset with features \(X\in\mathbb{R}^{n\times p}\) and targets \(y\in\mathbb{R}^n\), the linear model predicts \(\hat y = Xw + b\) with weights \(w\) and intercept \(b\). The base (least squares) loss is:  
\[
L_{\text{OLS}}(w,b) = \frac{1}{2n} \sum_{i=1}^n (y_i - (w^T x_i + b))^2.
\]  
Regularized losses add a penalty term on \(w\):

- **Ridge (L2) Regularization:**  
  \[
  L_{\text{ridge}}(w,b) 
  = \frac{1}{2n}\sum_i (y_i - (w^T x_i + b))^2 + \alpha \|w\|_2^2.
  \]  
  Here \(\|w\|_2^2 = \sum_j w_j^2\). The hyperparameter \(\alpha\ge0\) controls shrinkage (larger \(\alpha\) → more penalty)【3†L196-L204】【23†L313-L322】.  
  *This matches* the scikit-learn description: *“minimize \(\|Xw-y\|_2^2 + \alpha \|w\|_2^2\), where \(\alpha\) controls shrinkage”*【3†L196-L204】.

- **Lasso (L1) Regularization:**  
  \[
  L_{\text{lasso}}(w,b) 
  = \frac{1}{2n}\sum_i (y_i - (w^T x_i + b))^2 + \alpha \|w\|_1.
  \]  
  Here \(\|w\|_1 = \sum_j |w_j|\). Lasso adds an L1 penalty, which enforces sparsity by driving some coefficients to zero【7†L332-L337】【17†L239-L247】. It *“prefers solutions with fewer non-zero coefficients”*【7†L320-L327】. Larger \(\alpha\) means stronger penalty and more zeros【17†L239-L247】.

- **Elastic Net (L1+L2) Regularization:**  
  \[
  L_{\text{elastic}}(w,b) 
  = \frac{1}{2n}\sum_i (y_i - (w^T x_i + b))^2 
  + \alpha\rho\,\|w\|_1 + \frac{\alpha(1-\rho)}{2}\,\|w\|_2^2.
  \]  
  Here \(\rho\in[0,1]\) (called `l1_ratio` in sklearn) trades off L1 vs L2. Elastic Net combines both penalties【6†L617-L624】【23†L365-L373】. If \(\rho=1\), it’s equivalent to Lasso; if \(\rho=0\), it’s equivalent to Ridge【6†L619-L624】【12†L1030-L1033】. This hybrid is useful when features are highly correlated【6†L610-L618】【23†L365-L373】.

## Loss Functions with Regularization

In summary, the penalized loss functions are:

- **Ridge:** \(L = \text{MSE} + \alpha \sum_j w_j^2.\)  
- **Lasso:** \(L = \text{MSE} + \alpha \sum_j |w_j|.\)  
- **Elastic Net:** \(L = \text{MSE} + \alpha \bigl(\rho\sum_j |w_j| + \tfrac{1-\rho}{2}\sum_j w_j^2\bigr).\)

Each adds a term to the standard mean-squared error (MSE). Note that different sources may absorb factors of \(n\) or \(1/2\) differently, but the core idea is the extra \(\ell_2\) or \(\ell_1\) penalty on the coefficients. These penalties discourage large weights and effectively reduce model complexity【23†L313-L322】【17†L239-L247】.

## Solutions and Derivations

- **Ridge Closed-Form:** Setting \(\partial L/\partial w = 0\) for the Ridge loss gives the *normal equation*:  
  \[
  (X^T X + n\alpha I)w = X^T y.
  \]  
  Equivalently, \(w = (X^TX + \lambda I)^{-1}X^Ty\), where \(\lambda = n\alpha\) (or \(\lambda=2n\alpha\) depending on convention)【23†L313-L322】. This has a unique solution even when \(X^TX\) is singular (thanks to the \(\lambda I\) term)【22†L1-L4】【23†L342-L350】. In other words, Ridge shrinks the OLS solution by adding \(\alpha\) to the eigenvalues of \(X^TX\). 

- **Lasso (No Closed Form):** The L1 penalty is not differentiable at zero, so there is no simple closed-form solution. Instead, Lasso is typically solved via iterative methods like coordinate descent or LARS. Intuitively, each coefficient update solves a one-dimensional problem with a *soft-thresholding* step【7†L392-L400】. For each feature \(j\),  
  \[
  w_j \leftarrow S\!\Bigl(\frac{x_j^T (y - X_{-j}w_{-j})}{\|x_j\|_2^2}, \frac{\alpha}{\|x_j\|_2^2}\Bigr),
  \]  
  where \(S(z,\kappa) = \operatorname{sign}(z)\max(0,|z|-\kappa)\) is the soft-thresholding operator【7†L392-L400】. When \(|z|\le \kappa\), \(w_j\) is set to 0. This is why Lasso yields sparse solutions【7†L392-L400】【23†L344-L353】. 

- **Elastic Net (Iterative):** The Elastic Net loss is convex but also non-differentiable due to the L1 part. It is solved by coordinate descent or similar optimizers. One can also view Elastic Net as first augmenting the data and then doing Lasso on the augmented set (per Zou & Hastie, 2005). Crucially, Elastic Net inherits **sparsity** from Lasso and **grouping** from Ridge【6†L603-L611】【23†L365-L373】. Its optimality condition combines both penalties:  
  \[
  \nabla_w L 
  = -\frac{1}{n}X^T(y - Xw) + \alpha\bigl((1-\rho)w + \rho\,\text{sign}(w)\bigr) = 0,
  \]  
  where \(\text{sign}(w)\) is the element-wise sign (subgradient of \(\ell_1\)). There is no closed-form, but coordinate updates shrink each coefficient with both a ridge-term and a soft threshold.

## Sparsity and Feature Selection (Lasso)

A key property of Lasso is **sparsity**: some weights \(w_j\) become exactly zero, effectively removing the corresponding feature【7†L320-L327】【17†L239-L247】. This happens because the L1 penalty has “corners” at zero that the solution can hit. In practice:
- If a feature is not strongly correlated with the target (relative to \(\alpha\)), its coefficient is shrunk to 0.
- This leads to automatic feature selection. As IBM explains, *“larger λ... shrink[s] more coefficients towards zero; this reduces the importance of (or eliminates) some features”*【17†L239-L247】.
- Consequently, Lasso is popular in high-dimensional settings and compressed sensing【7†L320-L327】【17†L239-L247】. It simplifies the model and improves interpretability. In contrast, Ridge *never* sets weights exactly to zero; it only makes them small【23†L344-L353】.

## Elastic Net Trade-offs

Elastic Net aims to balance Lasso and Ridge:
- When features are **highly correlated**, Lasso tends to pick one and ignore the others, whereas Elastic Net tends to keep them together (grouping effect)【6†L610-L618】. 
- Setting the L1 ratio \(\rho\) closer to 1 yields more sparsity (like Lasso), while \(\rho\) closer to 0 acts more like Ridge【6†L603-L611】【12†L1030-L1033】.
- Empirically, Elastic Net often performs better when \(p \gg n\) or when there are many redundant features, because the L2 part adds stability to the solution. As scikit-learn notes, *“a practical advantage of trading off between Lasso and Ridge is that Elastic-Net inherits some of Ridge’s stability”*【6†L603-L611】.
- Tuning \(\alpha\) and \(\rho\) (via cross-validation) is crucial. Elastic Net adds complexity (two hyperparameters) but can outperform pure Lasso or Ridge when tuned well【6†L603-L611】【12†L1030-L1033】.

## Hyperparameters: α and l1_ratio

- **\(\alpha\)** (regularization strength): Applies to Ridge, Lasso, and Elastic Net (for Elastic Net it scales both penalties). In sklearn, a larger \(\alpha\) means more regularization (tighter penalty). Very large \(\alpha\) forces all \(w_j\) toward zero (overshrink), while \(\alpha=0\) recovers OLS. Choosing \(\alpha\) is done via cross-validation (e.g. `RidgeCV`, `LassoCV`, `ElasticNetCV` in sklearn). The IBM guides note that “\(\lambda\) (λ) balances bias and variance”【17†L278-L287】.

- **\(\rho\) (l1_ratio):** Only in Elastic Net. \(\rho=1\) reduces to Lasso (pure L1), \(\rho=0\) reduces to Ridge (pure L2)【6†L619-L624】【12†L1030-L1033】. Values between 0 and 1 give a mixture. Typical practice: search over a grid of \(\alpha\) and \(\rho\) with cross-validation, or use built-in CV solvers.



## Comparison of Methods

| Method      | Penalty           | Sparsity | When to Use                          | Pros                                            | Cons                                           |
|-------------|-------------------|----------|--------------------------------------|-------------------------------------------------|------------------------------------------------|
| **OLS**     | None              | No       | Baseline, low-dimensional, large data | Unbiased; no shrinkage                        | Overfits easily; unstable with correlated features |
| **Ridge**   | L2 (squared)      | No       | Collinear features; \(p>n\)           | Stabilizes estimates; handles multicollinearity【10†L548-L554】【23†L313-L322】 | No feature elimination; all coefficients remain non-zero【23†L344-L353】 |
| **Lasso**   | L1 (absolute)     | Yes      | Feature selection; high-dimensional   | Produces sparse model (auto feature selection)【17†L239-L247】【23†L344-L353】 | Can arbitrarily pick one among correlated features; optimization is slower |
| **ElasticNet** | L1 + L2 (mix)  | Partial  | Grouped/Correlated features; \(p\gg n\) | Balances sparsity and stability; groups correlated features【6†L603-L611】【23†L365-L373】 | Two hyperparameters (\(\alpha,\rho\)); more complex to tune |

- **Penalties:** OLS has no penalty. Ridge adds \(\|w\|_2^2\), Lasso adds \(\|w\|_1\), ElasticNet adds both.
- **Sparsity:** Only Lasso (and ElasticNet partially) set coefficients to exactly zero【17†L239-L247】【23†L344-L353】.
- **Use Cases:** If you suspect many irrelevant features or need feature selection, use Lasso or ElasticNet. If features are highly correlated and you want to keep them, ElasticNet or Ridge is preferred【6†L610-L618】【23†L365-L373】. For a quick fix to overfitting without worrying about sparsity, use Ridge【10†L548-L554】.
- **Pros/Cons:** Ridge is simple and convex (closed form) but doesn’t reduce feature count. Lasso yields interpretability but can be unstable in some cases. ElasticNet is versatile but adds complexity.

## Conclusion

Regularization is essential for controlling model complexity in regression. Ridge (L2) and Lasso (L1) are two cornerstone methods:

- **Ridge:** Improves generalization by shrinking weights; reduces variance【23†L313-L322】【10†L548-L554】. It is simple (closed-form solution) but does not perform feature selection【23†L344-L353】.
- **Lasso:** Encourages sparsity, performing feature selection automatically【17†L239-L247】【7†L320-L327】. It is invaluable for high-dimensional data but requires iterative solvers.
- **Elastic Net:** Provides a middle-ground, useful when predictors are correlated【6†L610-L618】【23†L365-L373】. It introduces an extra parameter (`l1_ratio` or \(\rho\)) to balance L1 vs L2.

Understanding and tuning the hyperparameters (\(\alpha\) and \(\rho\)) is key. Typically one uses cross-validation to select them for the best bias-variance tradeoff. In practice, ElasticNet or LassoCV/RidgeCV in scikit-learn can automate this process.

---

## References

- Scikit-learn User Guide – *Linear Models (Ridge, Lasso, ElasticNet)*【3†L196-L204】【7†L320-L328】【6†L603-L611】.  
- Scikit-learn Example – *Ridge vs OLS overview and plots*【10†L548-L554】.  
- IBM Knowledge Center – *What is Ridge Regression?*【23†L233-L241】【23†L313-L322】.  
- IBM Knowledge Center – *What is Lasso Regression?*【17†L239-L247】【17†L251-L259】.  
- IBM Documentation – *Elastic Net Regression (Python sklearn)*【21†L5-L9】.  

