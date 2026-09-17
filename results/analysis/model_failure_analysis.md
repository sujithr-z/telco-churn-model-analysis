# Failure Analysis: Where Logistic Regression Breaks

**Model Evaluated:** Logistic Regression ($C=1.0$, $\ell_2$ penalty, solver=`lbfgs`)  
**Dataset Split:** Validation Partition ($N = 1,402$, Churn Rate = 26.46%)  
**Decision Threshold:** $\tau = 0.50$ (Default)

---

## 1. Validation Error Summary

At the standard decision cutoff of $\tau = 0.50$, the validation classification results are:

| Metric / Outcome | Count / Value | Formula / Description |
| :--- | :---: | :--- |
| **Total Validation Samples** | `1,402` | 100% of validation set |
| **True Positives (TP)** | `199` | Correctly identified churners |
| **True Negatives (TN)** | `920` | Correctly identified retained customers |
| **False Positives (FP)** | `111` | Retained customers falsely flagged as churners |
| **False Negatives (FN)** | `172` | **Missed churners** who left without detection |
| **Accuracy** | `79.81%` | $\frac{TP + TN}{Total} = \frac{1119}{1402}$ |
| **Precision** | `64.19%` | $\frac{TP}{TP + FP} = \frac{199}{310}$ |
| **Recall** | `53.64%` | $\frac{TP}{TP + FN} = \frac{199}{371}$ |
| **F1-Score** | `0.5844` | Harmonic mean of Precision and Recall |
| **ROC-AUC** | `0.8403` | Overall ranking / discrimination ability |
| **PR-AUC (Average Precision)** | `0.6607` | Baseline random chance = 0.2646 |

```text
Confusion Matrix (Validation, tau = 0.50):
                         Predicted Retained (0)    Predicted Churn (1)
Actual Retained (0)            TN = 920                  FP = 111
Actual Churn (1)               FN = 172                  TP = 199
```

> **Takeaway:** While overall accuracy is ~80%, the model suffers from an acute **False Negative Rate of 46.36%** (missing 172 out of 371 actual churners), exposing the telecommunications provider to severe unmitigated customer attrition.

---

## 2. False Negative Analysis (Missed Churners: $FN = 172$)

### Who is being missed?
False Negatives represent customers who cancelled their subscriptions, but the model assigned them a churn probability $p < 0.50$.

1. **Long-Contract Churners are Universally Missed**:
   - **One-year contracts:** Actual churners = 31, FN = 28 (**FN Rate = 100.0%**).
   - **Two-year contracts:** Actual churners = 10, FN = 10 (**FN Rate = 100.0%**).
   - *Why?* The linear model assigns massive negative weights to annual contracts (`Contract_One year` = -0.61, `Contract_Two year` = -1.25). Because the linear decision boundary cannot represent complex conditional interactions, having an annual contract creates an insurmountable negative bias in logit $z = w^T x + b$, rendering the model blind to churners with long-term contracts.

2. **High-Tenure Churners are Blind Spots**:
   - **Tenure 49+ months:** Actual churners = 31, FN = 28 (**FN Rate = 97.8%**).
   - **Tenure 25–48 months:** Actual churners = 41, FN = 26 (**FN Rate = 63.4%**).
   - In contrast, **Tenure 0–6 months** has an FN Rate of only **29.7%**.
   - *Why?* `tenure` is the single strongest negative weight in the model ($w = -1.5153$). A long-tenured customer who experiences sudden service dissatisfaction (e.g. price increase, network failure) is masked by their historical loyalty score.

3. **DSL and Non-Fiber Churners**:
   - **DSL Internet Churners:** FN Rate = **74.0%** (51 missed out of 90).
   - **No Internet Service Churners:** FN Rate = **100.0%** (17 missed out of 22).
   - In contrast, **Fiber Optic Churners** have an FN Rate of **31.2%** (104 missed out of 259).
   - *Why?* Fiber optic carries a heavy positive linear weight ($w = +0.8278$). Customers on lower-tier DSL or standalone landline services lack this risk booster and frequently fall beneath the 0.50 cutoff.

### False Negative Probability Breakdown (Near-Miss vs. Distant Miss)
| Probability Range | FN Count | % of All FN | Cumulative % | Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **$0.45 \le p < 0.50$** | `23` | `13.4%` | `13.4%` | **Immediate Borderline**: Recoverable with slight threshold reduction |
| **$0.40 \le p < 0.45$** | `25` | `14.5%` | `27.9%` | **Near-Threshold**: Recoverable at $\tau = 0.40$ |
| **$0.30 \le p < 0.40$** | `31` | `18.0%` | `45.9%` | **Moderate Uncertainty**: Recoverable at $\tau = 0.30$ |
| **$p < 0.30$** | `93` | `54.1%` | `100.0%` | **Structural Blindspots**: Unreachable by threshold tuning alone |

> **Key Finding:** **45.9% of all False Negatives** ($n = 79$) have probabilities between $0.30$ and $0.50$. Lowering the threshold from $0.50 \to 0.30$ directly captures over 45% of currently missed churners. However, **54.1% of FN cases** ($n = 93$) have $p < 0.30$; these represent structural limitations of linear boundaries (e.g. high-tenure, multi-year contracts).

---

## 3. False Positive Analysis (False Alarms: $FP = 111$)

### Who is being falsely flagged?
False Positives are loyal, non-churning customers ($y = 0$) whom the model predicted would churn ($p \ge 0.50$).

1. **Short-Tenure, Month-to-Month Retained Customers**:
   - **Tenure 0–6 months Non-Churners:** Actual Non-Churners = 143, FP = 51 (**FP Rate = 28.2%**).
   - In contrast, **Tenure 49+ months Non-Churners:** FP Rate = **0.0%** (only 3 false alarms out of 381).
   - *Why?* Newer customers naturally have low tenure ($w = -1.52$ is absent) and month-to-month contracts ($w = 0$ baseline). If they subscribe to Fiber optic or Electronic checks, their probability instantly shoots above 0.50 despite having no intention of leaving.

2. **Fiber Optic & Electronic Check Users**:
   - **Fiber Optic Non-Churners:** Actual Non-Churners = 359, FP = 87 (**FP Rate = 25.8%**).
   - **Electronic Check Non-Churners:** Actual Non-Churners = 269, FP = 66 (**FP Rate = 32.1%**).
   - In contrast, **Credit Card / Bank Transfer Non-Churners:** FP Rate = **4.3%** (only 17 FP out of 296).
   - *Why?* Fiber optic and Electronic checks are the two strongest positive categorical predictors. Non-churners with these two attributes get heavily penalized by the additive linear logit.

3. **High Monthly Spend with Moderate Tenure**:
   - **Monthly Charges > $90 Non-Churners:** FP Rate = **20.4%** (43 FP out of 211).
   - High monthly fees combined with paperless billing push otherwise stable customers over the decision boundary.

### False Positive Probability Breakdown
| Probability Range | FP Count | % of All FP | Cumulative % | Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **$0.50 \le p < 0.55$** | `32` | `28.8%` | `28.8%` | **Marginal False Alarm**: Just over the 0.50 boundary |
| **$0.55 \le p < 0.60$** | `31` | `27.9%` | `56.8%` | **Moderate False Alarm**: Modest linear elevation |
| **$0.60 \le p < 0.70$** | `37` | `33.3%` | `90.1%` | **Strong False Alarm**: Confluence of multiple positive weights |
| **$p \ge 0.70$** | `11` | `9.9%` | `100.0%` | **High-Confidence False Alarm**: Extreme alignment of risk features |

> **Key Finding:** Over **56.8% of False Positives** ($n = 63$) reside in the narrow $[0.50, 0.60)$ band, demonstrating that many false alarms are borderline cases caused by rigid additive thresholds.

---

## 4. Main Failure Patterns (Empirical Evidence)

### Pattern 1: The "Long-Contract Blindspot" (Extreme False Negative Concentration)
- **Evidence:** FN rate is **100.0%** (10/10) for Two-Year contracts and **100.0%** (28/31) for One-Year contracts, compared to **40.2%** (134/330) for Month-to-Month contracts.
- **Interpretation:** In the fitted Logistic Regression model, long contract terms have massive negative coefficients (`Contract_Two year` = -1.25). The linear model mathematically cannot flag an annual-contract customer as churn unless virtually all other 28 features are maximally positive. When long-term contract customers do churn (e.g. end-of-term attrition, service breakdown), the model misses them almost 100% of the time.

### Pattern 2: The "High-Tenure Immunity Trap"
- **Evidence:** FN rate escalates monotonically with customer tenure:
  - `0–6 months`: **29.7%** FN rate (16 missed / 164 churners)
  - `7–12 months`: **43.1%** FN rate (22 missed / 51 churners)
  - `13–24 months`: **53.7%** FN rate (29 missed / 54 churners)
  - `25–48 months`: **63.4%** FN rate (26 missed / 41 churners)
  - `49+ months`: **97.8%** FN rate (28 missed / 31 churners)
- **Interpretation:** Because `tenure` enters the model as a strictly linear negative term ($w = -1.5153$), long-tenured customers receive an enormous "protective" negative logit. Logistic regression treats tenure as continuous armor against churn, failing to capture late-tenure dissatisfaction.

### Pattern 3: The "New Customer / Fiber Optic / Electronic Check False Alarm"
- **Evidence:** Retained customers with `tenure 0–6 months`, `Fiber optic`, and `Electronic check` suffer an FP rate of **28.2%** (51 false alarms out of 143 non-churners), compared to only **0.0%** (3 / 381) in the 49+ month band.
- **Interpretation:** The model adds positive weights for lack of tenure ($+0$), Fiber optic ($+0.83$), and Electronic check ($+0.32$). For a newly onboarded customer, this automatic summation breaches the 0.50 threshold before the customer has established regular payment habits.

### Pattern 4: DSL & Standalone Landline Churn Invisibility
- **Evidence:** Churners with DSL have an FN rate of **74.0%** (51/90) and No Internet Service have an FN rate of **100.0%** (17/22), compared to **31.2%** (104/259) for Fiber optic.
- **Interpretation:** Because `InternetService_Fiber optic` is the primary positive risk indicator, customers churning from legacy DSL or landline services lack strong positive indicators and pass completely undetected.

---

## 5. What This Means for the Model

The systematic errors observed are not random noise; they stem directly from the **mathematical constraints of Logistic Regression**:

1. **Inability to Model Feature Interactions (Linear Additivity)**:
   - Logistic regression computes $z = \sum w_i x_i + b$. It assumes the effect of `tenure` is identical regardless of whether a customer has a `Month-to-month` or `Two year` contract.
   - In reality, churn behavior is highly interactive: a price hike on a 48-month fiber customer behaves very differently from a price hike on a 2-month customer. Tree-based models (e.g. XGBoost, Random Forest) or explicit interaction terms ($x_i \cdot x_j$) are required to capture these dynamics.

2. **Rigid Linear Decision Boundary**:
   - Logistic regression separates classes with a single 29-dimensional hyperplane. Customer churn risk is non-monotonic and multimodal (e.g., risk is high in months 1–3, drops in months 4–24, and spikes again around contract expiration in month 24). A single linear slope cannot fit this U-shaped retention hazard curve.

3. **Sub-Optimal Decision Threshold ($\tau = 0.50$) on Imbalanced Data**:
   - With an unweighted baseline churn rate of 26.5%, the default 0.50 threshold forces the model to require overwhelming evidence ($> 50\%$ certainty) before raising an alarm.
   - Adjusting $\tau \to 0.25$ aligns the decision cutoff with empirical risk distributions, recovering 45.9% of borderline False Negatives.

---

## 6. ACM One-Page Write-Up Material

### Where the model breaks

> **Logistic Regression exhibits two primary failure modes driven by its linear additive structure: high False Negatives among long-tenure/contracted customers, and high False Positives among newly onboarded digital subscribers.** Because the model assigns large negative weights to contract duration (`Contract_Two year` $w = -1.25$) and customer tenure ($w = -1.52$), long-tenured customers who churn are almost universally missed (FN rate = 100.0% on two-year contracts; 97.8% on 49+ month tenure). The model treats tenure as perpetual loyalty, creating an unyielding protective bias that masks late-lifecycle attrition.
>
> Conversely, the model generates elevated false alarms on new subscribers (FP rate = 28.2% for 0–6 month tenure), where the absence of tenure combines additively with positive risk coefficients for `Fiber optic` ($+0.83$) and `Electronic check` ($+0.32$). Furthermore, legacy DSL and landline churners pass undetected (FN rate = 74.0%) due to the absence of the fiber risk multiplier. These findings demonstrate that while Logistic Regression achieves a strong ranking baseline (ROC-AUC = 0.8403), its linear decision boundary cannot capture non-monotonic retention hazards or multi-feature interactions without non-linear architectures or engineered interaction features.

---
