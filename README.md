# ML From Scratch

A NumPy-based machine learning library where I implement core machine learning algorithms from scratch, study the mathematics behind them, and gradually turn the implementations into reusable, tested components.

The main idea behind this project is simple:

> **I don't want to only know how to use a machine learning algorithm. I want to understand what is happening underneath it.**

So instead of starting with `scikit-learn` and treating the models as black boxes, I built the main components myself using Python and NumPy.

This project started as a learning exercise while studying machine learning, but I decided to structure it more like a small ML library. That meant adding things such as preprocessing, metrics, model selection, cross-validation, pipelines, testing, and benchmarks instead of keeping everything inside separate notebooks.

---

## Why I Built This

When learning machine learning, it is very easy to write something like:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

and get a model working without really understanding what happened between `fit()` and `predict()`.

I wanted to go one level deeper.

For every algorithm, the goal was to understand:

* What problem does it solve?
* What assumptions does it make?
* What is the objective/loss function?
* How are the parameters learned?
* What does the optimization process look like?
* What happens when the input is wrong?
* How should the implementation be tested?
* How does the implementation behave on real data?
* How close is it to a mature implementation such as scikit-learn?

The project therefore became a combination of **machine learning theory + NumPy implementation + software engineering**.

---

## What This Project Covers

The library currently covers several parts of the classical machine learning workflow:

* Supervised learning
* Neural networks
* Decision trees
* Ensemble learning
* Unsupervised learning
* Data preprocessing
* Model evaluation
* Cross-validation
* Hyperparameter search
* Pipelines
* Automated testing
* Benchmarking
* Real-world dataset experiments

The implementations are intentionally educational. The purpose is understanding and engineering practice, not replacing production ML libraries.

---

## Project Structure

```text
ML from scratch/
│
├── benchmarks/
│   └── benchmark_linear_regression.py
│
├── notebooks/
│   ├── 01_linear_regression_real_world.ipynb
│   ├── 02_logistic_regression_real_world.ipynb
│   ├── 03_neural_network_real_world.ipynb
│   └── 04_gradient_boosting_real_world.ipynb
│
├── src/
│   └── ml_from_scratch/
│       │
│       ├── __init__.py
│       ├── base.py
│       │
│       ├── linear_models/
│       │   ├── __init__.py
│       │   ├── linear_regression.py
│       │   └── logistic_regression.py
│       │
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── classification.py
│       │   └── regression.py
│       │
│       ├── neural_networks/
│       │   └── __init__.py
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── encoder.py
│       │   ├── scaler.py
│       │   └── splitting.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── pipeline.py
│       │
│       ├── model_selection/
│       │   ├── __init__.py
│       │   ├── cross_validation.py
│       │   ├── model_selection.py
│       │   └── grid_search.py
│       │
│       ├── trees/
│       │   ├── __init__.py
│       │   ├── boosting.py
│       │   ├── decision_tree.py
│       │   ├── decision_tree_regressor.py
│       │   ├── gradient_boosting.py
│       │   └── random_forest.py
│       │
│       └── decomposition/
│           ├── __init__.py
│           └── pca.py
│
└── tests/
```

`__pycache__`, virtual environments, test caches, and other local/generated files are excluded through `.gitignore`.

---

# Implemented Algorithms

## 1. Linear Regression

The first model in the project was Linear Regression.

The implementation uses **Batch Gradient Descent** instead of relying on a closed-form solver.

The main pieces include:

* Mean Squared Error
* Cost function
* Gradient computation
* Parameter updates
* Learning rate
* Iteration control
* Validation
* Early stopping
* Prediction
* R² scoring
* Gradient checking

The implementation was tested on the California Housing dataset and compared against scikit-learn.

### Example result

On the same train/test split:

| Metric | From Scratch | scikit-learn |
| ------ | -----------: | -----------: |
| MSE    |   110.907847 |   110.906097 |
| RMSE   |    10.531279 |    10.531196 |
| MAE    |     8.535128 |     8.535104 |
| R²     |     0.971054 |     0.971055 |

The prediction difference between the two implementations was very small.

The custom implementation was slower, which is expected because this version uses iterative gradient descent while the reference implementation uses a more optimized solver.

The benchmark was not about trying to beat scikit-learn.

It was mainly a sanity check that the implementation behaves correctly.

---

## 2. Logistic Regression

Logistic Regression was implemented from scratch using NumPy.

The implementation includes:

* Sigmoid activation
* Binary Cross-Entropy loss
* Gradient computation
* Batch Gradient Descent
* Probability prediction
* Class prediction
* Configurable classification threshold
* Input validation

The classification metrics were also implemented separately.

For the Breast Cancer Wisconsin dataset, I experimented with different decision thresholds rather than automatically assuming that `0.5` is always the correct choice.

For example, changing the threshold to `0.3` increased recall on the test split, which is useful for understanding the practical effect of the classification threshold.

This was one of the points where the project moved from simply implementing an algorithm to thinking about **how model outputs are actually used**.

---

# 3. Neural Networks

The neural network implementation was built from basic components rather than using an existing deep learning framework.

The implementation includes:

* Dense layers
* ReLU
* Sigmoid
* Tanh
* Forward propagation
* Backpropagation
* Parameter updates
* MSE loss
* Binary Cross-Entropy loss
* Prediction
* Gradient checking

The model was also tested on the Breast Cancer dataset and compared with scikit-learn's MLP implementation.

One of the useful parts of this stage was gradient checking.

Instead of assuming that the backpropagation equations were correct, numerical gradients were compared with the analytical gradients.

That made debugging the neural network considerably easier.

---

# 4. Decision Tree Classifier

The Decision Tree implementation was built recursively.

The main concepts implemented include:

* Entropy
* Information Gain
* Feature splitting
* Recursive tree construction
* Stopping criteria
* Maximum depth
* Minimum samples per split
* Prediction
* Random feature selection

The goal here was not to create a production-grade tree implementation.

It was to understand what actually happens when a decision tree searches for useful splits.

The implementation was tested against the Breast Cancer dataset and compared with scikit-learn.

---

# 5. Random Forest

Random Forest was implemented on top of the decision tree implementation.

The implementation uses:

* Bootstrap sampling
* Multiple decision trees
* Random feature selection
* Aggregation across trees
* Classification predictions
* Probability predictions
* Reproducible randomness

The important part for me here was understanding how Random Forest extends the idea of a single decision tree instead of treating it as a completely separate algorithm.

---

# 6. Gradient Boosting

Gradient Boosting was implemented together with a regression decision tree.

The simplified implementation follows the basic boosting idea:

1. Start with an initial prediction.
2. Calculate the residuals.
3. Train a weak regression tree on those residuals.
4. Update the current predictions.
5. Repeat for multiple estimators.

The implementation supports parameters such as:

* `n_estimators`
* `learning_rate`
* `max_depth`
* `min_samples_split`

This is intentionally a **simplified educational Gradient Boosting implementation**.

It is not intended to be a reimplementation of XGBoost, LightGBM, or CatBoost.

That distinction matters because production boosting libraries contain many additional optimization and regularization techniques that are outside the scope of this project.

---

# Unsupervised Learning

## 7. K-Means

K-Means was implemented using NumPy.

The implementation covers:

* Centroid initialization
* Distance calculation
* Cluster assignment
* Centroid updates
* Empty-cluster handling
* Convergence based on centroid movement
* Inertia
* Prediction
* `fit_predict()`

The implementation was kept intentionally simple so that the algorithm remains easy to follow.

I did not try to reproduce every optimization used by mature clustering libraries.

---

## 8. Principal Component Analysis

PCA was implemented using the covariance matrix and eigenvalue decomposition.

The implementation includes:

* Feature centering
* Covariance matrix calculation
* Eigenvalue decomposition
* Sorting principal components
* Selecting `n_components`
* Explained variance
* Explained variance ratio
* Transformation into the principal component space
* `fit_transform()`

The implementation also validates the number of components against the number of samples and features.

---

# Preprocessing

A machine learning model is only part of the workflow.

The project therefore includes basic preprocessing components.

## StandardScaler

A NumPy-based scaler was implemented for feature standardization.

The important part here is not only the formula itself, but **where fitting happens**.

For example, when using a train/validation split, the scaler must be fitted on the training data rather than on the entire dataset.

Otherwise information from the validation/test set can leak into the training process.

---

## OneHotEncoder

A simple categorical encoder was implemented with support for:

```python
handle_unknown="error"
```

and:

```python
handle_unknown="ignore"
```

One of the tests specifically checks that categories appearing only in validation data are not learned during training.

This was useful for demonstrating a small but important source of data leakage.

---

## Train/Test Split

The project also includes a NumPy-based train/test splitting utility with support for:

* Test size
* Shuffling
* Random state
* Input validation

---

# Metrics

The metrics are kept separate from the model implementations.

## Regression

Currently imp
