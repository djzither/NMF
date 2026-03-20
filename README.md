# Non-Negative Matrix Factorization (NMF) Recommender System

## Overview

This project implements a **Non-Negative Matrix Factorization (NMF)** model from scratch using convex optimization and applies it to build a **collaborative filtering recommender system**.

NMF decomposes a non-negative data matrix ( V ) into two lower-rank matrices ( W ) and ( H ), uncovering latent structure in the data. This technique is widely used in recommendation systems, image processing, and clustering.

---

## Features

* Implemented NMF using **CVXPY** with alternating minimization
* Built a **recommender system** using collaborative filtering
* Compared custom implementation with **Scikit-learn’s NMF**
* Applied NMF to:

  * Synthetic recommendation data
  * Real-world **face image dataset** (feature extraction)
* Visualized learned **basis components** (e.g., facial features)

---

## Mathematical Formulation

We solve the optimization problem:

$$
\min_{W,H \ge 0} \|V - WH\|_F
$$

where:

* $V \in \mathbb{R}^{m \times n}$ is the data matrix  
* $W \in \mathbb{R}^{m \times r}$ contains basis components  
* $H \in \mathbb{R}^{r \times n}$ contains coefficients  

Because this problem is not jointly convex, we use **alternating minimization**:

1. Fix ( H ), solve for ( W )
2. Fix ( W ), solve for ( H )
3. Repeat until convergence

---

## Project Structure

```
NMF/
│── nmf.py                # NMF class implementation (CVXPY)
│── recommender.py       # Recommendation system logic
│── faces.py             # Face dataset processing
│── experiments.py       # Runs experiments / comparisons
│── README.md
```

---

## Results

### 1. Recommendation System

* Successfully predicts missing user preferences
* Demonstrates collaborative filtering behavior
* Reconstructed matrix fills in sparse data

### 2. Face Decomposition

* Learned interpretable **basis images** such as:

  * Hair patterns
  * Facial structure
  * Lighting variations
* Shows how NMF extracts meaningful visual features

---

## Technologies Used

* Python
* NumPy
* CVXPY
* Scikit-learn
* Matplotlib

---

## Example Usage

```python
from nmf import NMFRecommender

model = NMFRecommender(rank=3)
model.fit(V)

V_reconstructed = model.reconstruct()
```

---

## Key Takeaways

* NMF provides an interpretable alternative to methods like SVD
* Alternating optimization enables solving non-convex problems
* Matrix factorization is a powerful tool for:

  * Recommendation systems
  * Dimensionality reduction
  * Feature extraction

---

## Author

Derek J Robinson
