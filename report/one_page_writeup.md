# ACM SIG AI — Telco Customer Churn (One-Page Summary & Report)

A machine-learning project built for the **ACM SIG AI recruitment task** using the IBM Telco Customer Churn dataset.

---

## 1. How I Understand Machine Learning

The learning process is a mathematical approximation:

```text
Reality → Data → Clean + Structure → Numerical Representation 
        → Model Choice → Parameter Optimization → Prediction on Unseen Data
```

For Logistic Regression:
$$z = Xw + b, \quad p = \frac{1}{1+e^{-z}}$$

---

## 2. Key Dataset & Cleaning Statistics

- **Raw Customers:** 7,043
- **Duplicates Dropped:** 33
- **Cleaned Dataset:** 7,010 customers ($19$ features, $26.49\%$ baseline churn rate)
- **Stratified Partitioning:**
  - Train (60%): 4,206 samples (26.51% churn)
  - Validation (20%): 1,402 samples (26.46% churn)
  - Test (20%): 1,402 samples (26.46% churn, strictly held out)

---

## 3. Data Preprocessing Matrix

- **Numerical Features (3):** `tenure`, `MonthlyCharges`, `TotalCharges` $\to$ `StandardScaler` ($x' = \frac{x-\mu}{\sigma}$)
- **Categorical Features (15):** One-Hot Encoded with `drop='first'` / `drop='if_binary'`
- **Final Transformed Representation:**
  $$X_{\text{train}} \in \mathbb{R}^{4206 \times 30}, \quad X_{\text{val}} \in \mathbb{R}^{1402 \times 30}, \quad X_{\text{test}} \in \mathbb{R}^{1402 \times 30}$$

---

## 4. Model Comparison & Threshold Tuning Results

| Model / Experiment | Threshold ($\tau$) | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.50 | 0.8403 | 0.6607 | 79.81% | 64.19% | 53.64% | 0.5844 |
| **Logistic Regression (Threshold-Tuned)** | 0.28 | 0.8403 | 0.6607 | 74.32% | 51.97% | 78.17% | 0.6243 |
| **Gradient Boosting (Tuned)** | 0.50 | 0.8444 | 0.6685 | 80.17% | 66.86% | 48.79% | 0.5642 |
| **Gradient Boosting (Threshold-Tuned)** | 0.25 | **0.8444** | **0.6685** | 73.18% | 51.82% | **80.59%** | **0.6308** |

---

## 5. Summary Failure Mode Findings

1. **Long-Tenure / Contract Blindspots:**
   - Both models exhibit elevated False Negative rates on Two-Year contracts ($100\%$) and 49+ month tenure ($97.8\%$) at $\tau=0.50$.
   - This occurs because extreme long-tenure churners are statistically rare in historical records ($\approx 2.7\%$), driving empirical leaf/logit probabilities below $0.50$.
2. **Threshold Shift Solution:**
   - Shifting the threshold to $\tau \approx 0.25$ aligns the decision boundary with empirical churn risk, recovering **$45.9\%$ of missed churners** ($79/172$).
3. **Where Gradient Boosting Genuinely Wins:**
   - Cuts False Alarms (False Positives) on Month-to-Month contracts by **$4.6\%$** and Fiber Optic users by **$5.4\%$**.
   - Improves early-stage churn detection (0–6 month tenure FN rate drops from $29.7\% \to 23.0\%$).

---

## 6. Visual Outputs

- **Pipeline & Feature Representation:** `results/figures/data_numerical_representation.png`
- **Evaluation Overview:** `results/figures/logistic_regression_evaluation.png`
- **Contract Error Rates:** `results/figures/error_by_contract.png`
- **Tenure Error Rates:** `results/figures/error_by_tenure.png`
- **Internet Service Error Rates:** `results/figures/error_by_internet_service.png`
- **Probability Distribution:** `results/figures/probability_distribution_by_error_type.png`
- **Model Comparison Curves:** `results/figures/model_comparison_roc_pr_curves.png`
