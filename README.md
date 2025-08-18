# ML From Scratch

Implementing Machine Learning algorithms from scratch using **NumPy** and **pandas**.  
Inspired by **Stanford's CS229 (Autumn 2018)**, this repository is a hands-on journey into the **mathematical**, **statistical**, and **programmatic** foundations of modern machine learning.

## Project Goals

- Reinforce understanding of core ML algorithms by building them **step-by-step**.
- Avoid high-level libraries like **scikit-learn** to focus on the **actual mechanics**.
- Reproduce and extend concepts from **Stanford CS229** with **clear visualizations**.
- Provide **clean, modular, and well-documented notebooks** for each algorithm.

## Implemented Algorithms (so far)

### Linear Models

- **Linear Regression**
  - Normal Equation (closed-form solution)
  - Gradient Descent (Batch)
  - Stochastic Gradient Descent (SGD)
  - Newton–Raphson (Quadratic approximation where applicable)

- **Locally Weighted Regression (LWR)**
  - Distance-based weighting
  - Gaussian kernel weighting
  - Experiments with different bandwidth parameters

- **Logistic Regression** (Binary classification)
  - Gradient Descent (Batch)
  - Stochastic Gradient Descent (SGD)
  - Newton–Raphson / IRLS (Iteratively Reweighted Least Squares)
  - Discussion of Normal Equation applicability and limitations for non-linear loss

### Perceptron & Linear Separators

- **Perceptron**
  - Perceptron Learning Algorithm (PLA)
  - Convergence behavior on linearly separable datasets

### Generalized Linear Models (GLMs)

- **Softmax (Multinomial) Regression**
  - Cross-entropy loss with gradient updates

- **Poisson Regression**
  - Log-link function, likelihood, and gradient updates

### Gaussian Discriminant Analysis (GDA)
  - **Linear Discriminant Analysis (LDA)** — shared covariance
  - **Quadratic Discriminant Analysis (QDA)** — separate covariance per class
  - Parameter estimation (φ, μ, Σ)
  - Decision boundary visualization with contours
  - Model persistence with `pickle`

### Naive Bayes

- **Multinomial Naive Bayes** (text classification setting)
- Implementation with **Laplace smoothing**
- Word probability estimation & class priors
- Experiments on small text datasets (toy examples)
## Structure

Each algorithm’s directory contains:

- Mathematical derivations / notes (Markdown or Jupyter cells)
- From-scratch implementation (**NumPy** / **pandas** only)
- Experiments & visualizations (**matplotlib**)


