# Project Learning Log & Technical Reflection: Telco Customer Churn

**Author:** Sujith  
**Project:** ACM SIG AI — Telco Customer Churn Prediction  
**Date:** September 2026  

---

## 1. From "Magic Algorithm" to Mathematical Modeling

When I initially started this task, I viewed Machine Learning mostly as a collection of black-box libraries and self-learning algorithms where you feed raw data to a computer and it somehow magically figures everything out on its own.

Working through the Telco churn dataset from the ground up changed my perspective entirely:

> **Machine Learning is fundamentally mathematical modeling.** We take a complex real-world phenomenon (customer retention and attrition) and construct a mathematical representation of it so that optimization algorithms can find a structured mapping from input features to target decisions.

---

## 2. Converting the Real World into Geometry: Data Representation

A computer cannot directly understand concepts like *"Month-to-month contract"* or *"Fiber optic internet"*. To compute with them, we must map them into Euclidean space:

1. **Cleaning & Sanity Checks:** Remove non-predictive identifiers (`customerID`), convert distorted types (e.g. whitespace in `TotalCharges` coerced to float), and eliminate invalid records.
2. **Feature Separation:** Isolate target labels ($y \in \{0, 1\}$ for Churn) from explanatory features ($X$).
3. **Scaling Continuous Metrics:** Continuous variables like `tenure` and `MonthlyCharges` have disparate scales. We standardize them ($z = \frac{x - \mu}{\sigma}$) so features with large absolute magnitudes don't dominate optimization gradients.
4. **Categorical Encoding:** Using One-Hot / Binary indicator encoding, multi-state categories become orthogonal binary dimensions with dropped baseline levels to avoid collinearity.

Through this pipeline, our customer table is transformed into a real-valued matrix:

$$X \in \mathbb{R}^{n \times d}$$

For our training partition, this became:

$$X_{\text{train}} \in \mathbb{R}^{4206 \times 30}$$

Every customer is now a coordinate point in a 30-dimensional mathematical space.

```text
Real World Customer (Contract, Tenure, Spend)
                      │
                      ▼
         Mathematical Preprocessing
       (StandardScaler + OneHotEncoder)
                      │
                      ▼
            X ∈ ℝ^(4206 × 30) Vector Space
```

---

## 3. The Mechanics of Learning: Logistic Regression

Once data is in vector form, we choose a mathematical function to approximate the relationship between inputs $X$ and outcome $y$.

For Logistic Regression, the model assumes a linear log-odds structure:

$$z = Xw + b = \sum_{i=1}^{d} w_i x_i + b$$

where $w \in \mathbb{R}^{30}$ is the parameter weight vector and $b \in \mathbb{R}$ is the intercept (bias).

To convert the unbounded real score $z \in (-\infty, +\infty)$ into a calibrated probability $p \in [0, 1]$, we apply the **logistic sigmoid function**:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}} = P(\text{Churn} = 1 \mid x)$$

### What "Learning" Actually Means

Before training, we do not know the true parameters $w$ and $b$. The computer does not "think"; it executes an optimization loop:

```text
Initialize w, b → Compute z = Xw + b → Compute Probabilities p = σ(z) 
       → Evaluate Binary Cross-Entropy Loss L(w, b) 
       → Calculate Gradients ∇_w L, ∇_b L → Update Weights (w, b) → Repeat
```

The algorithm systematically adjusts $w$ and $b$ until the loss function is minimized.

---

## 4. The "Paper and Glass" Analogy

The intuition behind model fitting and generalization can be summarized with a simple mental model:

> Imagine drawing a complex line on a piece of paper. Then you place a transparent glass sheet over it and try to draw a line on the glass that traces the line on the paper as closely as possible.
>
> The **paper** is the historical training data.  
> The **glass line** is the parametric model trying to approximate reality.

```text
   Paper (Ground Truth / Data)        Glass (Parametric Model)
  ┌───────────────────────────┐      ┌───────────────────────────┐
  │   •   •    •    •   •     │  vs  │     ─────────────────     │
  │     •   •    •    •       │      │        (Approximation)    │
  └───────────────────────────┘      └───────────────────────────┘
```

The goal is **not** to trace every erratic smudge or noisy outlier (overfitting). The goal is to capture the underlying trajectory so that when you place the glass sheet over **new, unseen paper** (test data), the approximation remains accurate.

---

## 5. Experimental Discipline: Preventing Data Leakage

Because our objective is out-of-sample generalization, strict dataset partitioning is required:

- **Training Set (60%, $N = 4,206$):** Exclusively used to compute sample statistics ($\mu, \sigma$), build categorical vocabularies, and optimize parameters ($w, b$).
- **Validation Set (20%, $N = 1,402$):** Used for model selection, threshold calibration ($\tau$), and failure mode inspection.
- **Test Set (20%, $N = 1,402$):** Kept strictly sealed and untouched until final evaluation to serve as an honest proxy for production performance.

---

## 6. Models as Different Mathematical Hypotheses

Different machine learning models are simply different mathematical architectures for approximating the data manifold:

| Model | Mathematical Structure | Strengths | Limitations |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear additive hyperplane: $z = w^T x + b$ | Highly interpretable, fast, strong baseline ($\text{ROC-AUC} = 0.8403$). | Cannot model multi-feature interactions without manual cross-terms; treats tenure monotonically. |
| **Gradient Boosting** | Ensembles of shallow decision trees: $\sum f_m(x)$ | Naturally captures hierarchical non-linear splits and feature interactions ($\text{ROC-AUC} = 0.8444$). | Higher complexity; still bound by base-rate scarcity in small sub-populations. |

This explains why **no single model is universally "best"** (No Free Lunch Theorem). The mathematical structure of the model must match the structural properties of the problem. For tabular customer records, linear models and tree ensembles are appropriate; spatial architectures like CNNs would make little sense because tabular columns have no spatial locality or translation invariance.

---

## 7. Key Empirical Insights from Our Telco Project

1. **The Default Threshold ($0.50$) is Sub-optimal for Imbalanced Problems:**
   - With a baseline churn rate of $26.5\%$, the default cutoff missed $46.4\%$ of actual churners ($FN = 172$).
   - Lowering the decision threshold to **$\tau \approx 0.25$** recovered **$45.9\%$ of borderline missed churners**, boosting validation $F_1$ to **$0.6308$** and Recall to **$80.59\%$**.
2. **Distinguishing Error Counts from Error Rates:**
   - High churn volume in Month-to-Month contracts ($134$ misses) is a function of large sample size.
   - In contrast, Two-Year contracts had a **$100\%$ False Negative Rate** ($9/9$ missed), exposing a structural blindspot where low historical base rates mask individual attrition events.

---

## 8. Summary Checklist of Work Completed

- [x] **Exploratory Data Analysis & Categorical Profiling** (`src/data/categorical/categorical_table_Data.py`)
- [x] **Data Cleaning & Target Extraction** (`src/data/cleaner.py`)
- [x] **Stratified Leak-Free 60/20/20 Partitioning** (`src/data/splitter.py`)
- [x] **Reusable Preprocessing Pipeline** (`src/preprocessing/pipeline.py`)
- [x] **Logistic Regression Training & Weight Inspection** (`src/model/train.py`)
- [x] **Granular Failure Analysis & Uncertainty Quantification** (`src/evaluation/analysis.py`)
- [x] **Gradient Boosting Cross-Validation Hyperparameter Tuning** (`src/model/gradient_boosting.py`)
- [x] **Side-by-Side Model Comparison & Subgroup Delta Auditing** (`src/model/compare_models.py`)
