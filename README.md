# ACM SIG AI — Telco Customer Churn

A machine-learning project built for the **ACM SIG AI recruitment task** using the IBM Telco Customer Churn dataset.

The main idea of this project was not just to train a model, but to understand what is actually happening underneath the machine-learning pipeline.

---

## 1. How I Understand Machine Learning

The way I approached ML changed while doing this project.

At first, ML looked like:

```text
Data → Library → Model → Prediction
```

But after working through the problem, I understood it more as **mathematical approximation**.

We start with something from reality and represent it as data. Then we convert that data into a numerical form that a mathematical model can work with.

The learning process is roughly:

```text
Reality
   ↓
Data
   ↓
Clean + Structure
   ↓
Numerical Representation
   ↓
Choose a mathematical model
   ↓
Initial parameters
   ↓
Prediction
   ↓
Compare with the real target
   ↓
Calculate error / loss
   ↓
Update parameters
   ↓
Repeat
   ↓
Learned approximation
   ↓
Prediction on unseen data
```

The model is not learning "reality" itself. It is finding a mathematical approximation of the relationship that exists in the observed data.

For Logistic Regression, the core mathematical idea is:

$$z = Xw + b$$

where `X` is the feature matrix, `w` is the vector of learned parameters, and `b` is the bias.

The result is converted into a probability using the sigmoid function:

$$p = \frac{1}{1+e^{-z}}$$

So, at a high level, the model is finding parameter values that make its mathematical approximation agree with the training observations as well as possible.

> **Reality → data → mathematical representation → parameter estimation → approximation → prediction**

This was the main thing I took away from the project.

---

## 2. Dataset

The project uses the IBM Telco Customer Churn dataset.

The target variable is:

```text
Churn
Yes → 1
No  → 0
```

The dataset contains customer information such as:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `Contract`
- `InternetService`
- `PaymentMethod`
- `OnlineSecurity`
- `TechSupport`
- `StreamingTV`
- `SeniorCitizen`
- and other customer/service attributes.

The original dataset contains around 7,000 customers.

After cleaning:

```text
Customers:       7,010
Original features after removing customerID: 19
Overall churn:   26.49%
Churned:         1,857
Not churned:     5,153
```

---

## 3. Data Cleaning and Preparation

The first step was to make the raw table usable.

### Cleaning

The cleaning process included:

1. Loading the raw CSV.
2. Removing duplicate rows.
3. Handling blank/missing values in the dataset.
4. Removing `customerID` because it identifies a customer but does not provide useful predictive information.
5. Separating the input features from the target.

We removed:

```text
33 duplicate rows
```

leaving:

```text
7,010 customers
```

The important point is that cleaning is not just about making the table look nice. The goal is to make sure the information going into the model represents meaningful features rather than identifiers, duplicated observations, or invalid values.

---

## 4. Target and Feature Data

After cleaning, the data is separated into:

```text
X = input features
y = target
```

where:

```text
X → customer information
y → Churn
```

The data is then split into training, validation, and test sets.

```text
                Customers
                    │
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       Training  Validation   Test
        60%         20%       20%
       4206        1402       1402
```

The split is stratified so that the churn proportion remains approximately the same in each set.

```text
Train       26.51% churn
Validation  26.46% churn
Test        26.46% churn
```

The preprocessing is fitted only on the training data. Validation and test data are transformed using the already-fitted preprocessing steps.

This prevents information from the validation/test sets leaking into the training process.

---

## 5. Converting the Table into a Numerical Representation

A machine-learning model needs a numerical representation.

The original table contains different kinds of information, so the columns are separated into numerical and categorical features.

### Numerical features

Examples:

```text
tenure
MonthlyCharges
TotalCharges
```

These are standardized using `StandardScaler`.

Conceptually:

$$x' = \frac{x-\mu}{\sigma}$$

This puts numerical variables onto a comparable scale, which is particularly useful for Logistic Regression.

### Categorical features

Examples:

```text
Contract
InternetService
PaymentMethod
```

These cannot simply be treated as ordinary numbers because categories do not have a natural numerical ordering.

So they are converted using **one-hot encoding**.

For example:

```text
Contract

Month-to-month
One year
Two year
```

becomes separate numerical indicators.

After preprocessing:

```text
X_train shape = (4206, 30)
X_val shape   = (1402, 30)
X_test shape  = (1402, 30)
```

So the original customer table has now become a higher-dimensional numerical matrix.

---

## 6. Logistic Regression

The primary model is Logistic Regression.

The basic structure is:

```text
Customer features
       ↓
Numerical matrix X
       ↓
Linear combination
       ↓
z = Xw + b
       ↓
Sigmoid
       ↓
Churn probability
```

The model learns a parameter for each feature.

Some learned coefficients from the baseline model were:

```text
Positive:
InternetService_Fiber optic     +0.8278
TotalCharges                    +0.7589
PaperlessBilling_Yes            +0.3515
PaymentMethod_Electronic check  +0.3205
StreamingTV_Yes                 +0.2966

Negative:
OnlineSecurity_Yes              -0.3602
PhoneService_Yes                -0.5154
Contract_One year               -0.6093
Contract_Two year               -1.2548
tenure                          -1.5153
```

These coefficients describe how the fitted model uses the features. They should be interpreted as model associations, not as proof that a feature causes churn.

---

## 7. Validation Results

At the default classification threshold of `0.50`, the Logistic Regression model produced:

| Metric | Validation |
|---|---:|
| ROC-AUC | 0.8403 |
| PR-AUC | 0.6607 |
| Accuracy | 0.7981 |
| Precision | 0.6419 |
| Recall | 0.5364 |
| F1 | 0.5844 |

Confusion matrix:

```text
                    Predicted
                 No Churn   Churn
Actual
No Churn            920      111
Churn               172      199
```

So:

```text
TN = 920
FP = 111
FN = 172
TP = 199
```

Accuracy alone was not used as the main measure because the dataset is imbalanced: only about 26.5% of customers churn.

---

## 8. Threshold Experiment

The Logistic Regression model produces a probability first.

For example:

```text
Customer A → 0.12
Customer B → 0.37
Customer C → 0.81
```

A threshold converts that probability into a final class:

$$\hat y =
\begin{cases}
1 & p \geq \tau\\
0 & p < \tau
\end{cases}$$

At:

```text
τ = 0.50
```

we obtained:

```text
Precision = 64.2%
Recall    = 53.6%
F1        = 0.584
```

At:

```text
τ = 0.25
```

we obtained:

```text
Precision = 50.2%
Recall    = 82.2%
F1        = 0.624
```

So lowering the threshold catches more actual churners, but also creates more false positives.

The threshold of `0.25` produced the highest F1 among the tested thresholds. It is not a universal "optimal" threshold; the appropriate threshold depends on the cost of false positives versus false negatives.

---

## 9. Where the Model Breaks

The validation error analysis looked specifically at False Negatives and False Positives instead of only looking at the overall score.

At the 0.50 threshold:

```text
False Negatives = 172
False Positives = 111
```

Some strong subgroup patterns appeared.

### Long contracts and long tenure

The model missed many churners in some long-tenure/contract groups.

Observed validation error rates included:

```text
Two-year contract       FN rate = 100.0%
49+ months tenure       FN rate = 97.8%
```

These groups are relatively small, so the rates should be interpreted together with their sample sizes.

The result suggests that the model has difficulty identifying churn within these particular groups.

### New customers and internet service

Some false-positive patterns were also concentrated in particular groups:

```text
0–6 months tenure       FP rate = 28.2%
Fiber optic             FP rate = 25.8%
```

This means the model sometimes flags these customers as likely churners even when they ultimately do not churn.

### Probability overlap

The probability distribution also showed that many errors were not extreme.

Among the 172 false negatives:

```text
48 / 172 = 27.9%
```

had predicted probabilities between `0.40` and `0.50`.

Expanding the region:

```text
79 / 172 = 45.9%
```

had probabilities between `0.30` and `0.50`.

So a significant part of the error comes from customers sitting in an uncertain region rather than the model giving them an extremely low churn probability.

---

## 10. Gradient Boosting Comparison

A tuned Gradient Boosting model was also trained under the same experimental setup.

The validation ranking metrics were:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.8403 | 0.6607 |
| Gradient Boosting | 0.8444 | 0.6685 |

The improvement was relatively small.

The second model also did not completely remove the major subgroup error patterns.

For example:

```text
Two-year contract
Logistic Regression     100.0% FN
Gradient Boosting       100.0% FN

49+ months tenure
Logistic Regression      97.8% FN
Gradient Boosting       100.0% FN
```

This was useful because it showed that simply moving from a linear model to a nonlinear model does not automatically solve every problem in the data.

The comparison also showed that different models can make different mistakes on different customer groups.

---

## 11. What I Learned

The biggest thing I learned from this project was that machine learning is much less mysterious than I initially thought.

I initially saw ML as mostly using libraries and letting a machine "learn" something.

Now I see the library more as an implementation of mathematical machinery.

The actual process is:

```text
Reality
   ↓
Data
   ↓
Clean it
   ↓
Represent it numerically
   ↓
Choose a mathematical structure
   ↓
Estimate its parameters
   ↓
Make predictions
   ↓
Compare predictions with reality
   ↓
Measure the errors
```

Different ML models are basically different mathematical structures for approximating relationships in data.

Logistic Regression gives us a linear/additive mathematical structure.

Gradient Boosting uses decision trees and can represent relationships differently.

So the interesting part is not just:

> "Which library should I use?"

It is:

> "What is the mathematical structure of the problem, how should I represent the data, and what kind of approximation makes sense?"

This also made me realize why linear algebra, probability, multivariable calculus, optimization, and mathematical modelling are so useful for understanding ML.

---

## 12. Project Structure

```text
ACM-SIG-AI-TELCO-CHURN/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   └── splitter.py
│   │
│   ├── preprocessing/
│   │   └── pipeline.py
│   │
│   ├── model/
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── analysis.py
│   │
│   └── from_scratch/
│       └── logistic_regression.py
│
├── pipelines/
│   ├── data_pipeline.py
│   ├── training_pipeline.py
│   └── evaluation_pipeline.py
│
├── models/
│   ├── logistic_regression.pkl
│   └── preprocessor.pkl
│
├── results/
│   ├── analysis/
│   ├── figures/
│   └── predictions/
│
└── report/
    └── one_page_writeup.md
```

---

## 13. Visual Results

### Data → Numerical Representation

![Data to Numerical Feature Pipeline](results/figures/data_numerical_representation.png)

---

### Logistic Regression Evaluation

![Logistic Regression Evaluation](results/figures/logistic_regression_evaluation.png)

---

### Classification Error Analysis

![Error Rate by Contract Type](results/figures/error_by_contract.png)

---

### Error Across Tenure

![Error Rate Across Tenure Bands](results/figures/error_by_tenure.png)

---

### Error Across Internet Service

![Error Rate by Internet Service Type](results/figures/error_by_internet_service.png)

---

### Probability Distribution

![Probability Distribution by Classification Error Type](results/figures/probability_distribution_by_error_type.png)

---

### Logistic Regression vs Gradient Boosting

![Logistic Regression vs Gradient Boosting ROC & PR Curves](results/figures/model_comparison_roc_pr_curves.png)

---

## 14. Main Takeaway

This project started as a churn-classification task, but the main thing I took from it was the underlying idea of machine learning:

> **We take something from reality, represent it as data, convert that data into a mathematical form, choose a mathematical structure to approximate the relationship, estimate its parameters from observations, and then test how well that approximation works on unseen data.**

The model is not reality.

It is an approximation of reality.

The quality of that approximation depends on the data, the representation, the mathematical structure, the parameters, and how honestly we evaluate it.
