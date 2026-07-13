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

---

### Perceptron & Linear Separators

- **Perceptron**
  - Perceptron Learning Algorithm (PLA)
  - Convergence behavior on linearly separable datasets

---

### Generalized Linear Models (GLMs)

- **Softmax (Multinomial) Regression**
  - Cross-entropy loss with gradient updates

- **Poisson Regression**
  - Log-link function, likelihood, and gradient updates

---

### Probabilistic Models

- **Gaussian Discriminant Analysis (GDA)**
  - **Linear Discriminant Analysis (LDA)** — shared covariance
  - **Quadratic Discriminant Analysis (QDA)** — separate covariance per class
  - Parameter estimation (φ, μ, Σ)
  - Decision boundary visualization with contours
  - Model persistence with `pickle`

- **Multinomial Naive Bayes**
  - Laplace smoothing
  - Word probability estimation
  - Class prior estimation
  - Toy text classification experiments

---

### Unsupervised Learning

- **K-Means Clustering**
  - Lloyd's K-Means algorithm
  - Random & **K-Means++** initialization
  - Multiple random restarts (`n_init`)
  - Convergence using Inertia (Objective Function)
  - Voronoi decision boundary visualization
  - Model persistence with `pickle`
  - Validation against **scikit-learn**
  - Evaluation using **Inertia**, **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**

---

### Deep Learning

- **Artificial Neural Networks (ANN)**
  - Modular sequential neural network architecture
  - Fully Connected (Dense) layers
  - Regression, Binary Classification, and Multiclass Classification examples
  - Deep Autoencoder
  - Deep Denoising Autoencoder
  - Forward and Backpropagation from scratch
  - Mini-batch Gradient Descent
  - Weight initialization:
    - Random
    - Xavier (Glorot)
    - He (Kaiming)
  - Activation functions:
    - Linear
    - Sigmoid
    - Tanh
    - ReLU
    - Leaky ReLU
    - Softplus
    - Softmax
  - Loss functions:
    - Mean Squared Error (MSE)
    - Binary Cross-Entropy (BCE)
    - Categorical Cross-Entropy (CCE)
  - Optimizers:
    - SGD
    - Adagrad
    - RMSprop
    - Adam
    - AdamW
  - Dropout regularization
  - Model persistence using compressed NumPy (`.npz`)
  - Evaluation using regression, classification, and reconstruction metrics

---

## Structure

Each algorithm's directory contains:

- Mathematical derivations / notes (Markdown or Jupyter cells)
- From-scratch implementation (**NumPy** / **pandas** only)
- Experiments & visualizations (**matplotlib**)
- Model saving/loading using `pickle` or compressed NumPy (`.npz`)


