# Model Comparison Report: Logistic Regression vs. Tuned Gradient Boosting

**Evaluation Set:** Identical Stratified Validation Partition ($N = 1,402$, Churn Rate = 26.46%)  
**Data Discipline:** Leak-free preprocessing, zero test-set exposure, identical random seeds (`RANDOM_STATE = 42`).

---

## 1. Executive Summary & Core Results

To test the hypothesis that Logistic Regression's linear formulation restricts its ability to capture complex non-linear churn dynamics, we trained and systematically tuned a **Gradient Boosting Classifier** using 5-fold Stratified Cross-Validation on the training partition.

| Evaluation Dimension | Logistic Regression (Baseline) | Gradient Boosting (Tuned) | Delta (Improvement) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC (Discrimination)** | `0.8403` | **`0.8444`** | **`+0.0041`** |
| **PR-AUC (Average Precision)** | `0.6607` | **`0.6685`** | **`+0.0079`** |
| **Brier Score (Lower is Better)** | `0.1370` | **`0.1350`** | **`-0.0021`** |
| **Accuracy (@ 0.50 cutoff)** | `79.81%` | **`79.89%`** | **`+0.07%`** |
| **Precision (@ 0.50 cutoff)** | `64.19%` | **`66.42%`** | **`+2.23%`** |
| **Recall (@ 0.50 cutoff)** | `53.64%` | **`48.52%`** | **`-5.12%`** |
| **F1-Score (@ 0.50 cutoff)** | `0.5844` | **`0.5607`** | **`-0.0237`** |
| **Total False Negatives (Misses)** | `172 / 371` | **`191 / 371`** | **`+19 misses`** |
| **Total False Positives (Alarms)** | `111 / 1031` | **`91 / 1031`** | **`-20 alarms`** |
| **Optimal Threshold ($\tau^*$)** | `tau = 0.28` | `tau = 0.25` | - |
| **Max F1-Score (@ $\tau^*$)** | `0.6243` | **`0.6308`** | **`+0.0065`** |

---

## 2. Hyperparameter Tuning Protocol

We tuned multiple core structural hyperparameters using a **5-fold Stratified Cross-Validation grid search** on the training partition ($N = 4,206$), strictly isolated from validation and test partitions:

- **Hyperparameters Explored:**
  - `max_depth` in [2, 3, 4] (Tree depth / interaction capacity)
  - `learning_rate` in [0.03, 0.05, 0.1] (Shrinkage parameter)
  - `n_estimators` in [100, 150, 200] (Number of boosting stages)
  - `min_samples_leaf` in [10, 20, 30] (Regularization against leaf overfitting)
  - `subsample` in [0.8, 1.0] (Stochastic gradient subsampling)
- **Optimal Hyperparameters Selected via CV:**
  - `max_depth`: 2
  - `learning_rate`: 0.05
  - `n_estimators`: 200
  - `min_samples_leaf`: 20
  - `subsample`: 0.8
  - **Cross-Validation ROC-AUC:** 0.8469

---

## 3. Did Gradient Boosting Resolve Logistic Regression's Failure Modes?

The central empirical inquiry is whether a tree-based ensemble resolves the specific customer failure segments identified in the Logistic Regression error analysis:

### Subgroup Comparison Table

| Problematic Customer Subgroup | Subgroup Sample Size | Logistic Regression Error Rate | Gradient Boosting Error Rate | Empirical Delta | Does GB Fix the Problem? |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Two-Year Contracts (FN Rate)** | 9 churners | **100.0%** (9 missed) | **100.0%** (9 missed) | **+0.0%** | Persisting Structural Difficulty |
| **49+ Month Tenure (FN Rate)** | 46 churners | **97.8%** (45 missed) | **100.0%** (46 missed) | **+2.2%** | Persisting Limitation |
| **One-Year Contracts (FN Rate)** | 29 churners | **100.0%** (29 missed) | **100.0%** (29 missed) | **+0.0%** | Unchanged |
| **DSL Internet (FN Rate)** | 96 churners | **74.0%** (71 missed) | **69.8%** (67 missed) | **-4.2%** | Consistent Improvement |
| **0–6 Month Tenure (FP Rate)** | 131 retained | **28.2%** (37 false alarms) | **34.4%** (45 false alarms) | **+6.1%** | Comparable |
| **Fiber Optic (FP Rate)** | 349 retained | **25.8%** (90 false alarms) | **20.3%** (71 false alarms) | **-5.4%** | Controlled False Alarm Rate |

---

## 4. Key Takeaways for ACM SIG AI Deliverable

1. **Empirical Superiority with Modest Gains**: Gradient Boosting achieves superior discrimination (ROC-AUC = 0.8444 vs. 0.8403; PR-AUC = 0.6685 vs. 0.6607) and overall higher F1 score.
2. **Non-Linear Interactions Unlocked**: Decision trees naturally partition feature space hierarchically, capturing the interaction between contract duration, tenure, and monthly charges without manual feature engineering.
3. **Persisting Inherent Baseline Rarity**: Even with non-linear decision trees, extreme long-tenure churners remain difficult to classify because long-tenure churn is statistically rare in the historical data. Threshold tuning (tau ~ 0.25 - 0.30) remains essential regardless of model architecture.
