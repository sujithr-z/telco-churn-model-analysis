# ACM SIG AI — Telco Customer Churn

A machine-learning project built for the **ACM SIG AI recruitment task** using the IBM Telco Customer Churn dataset.
What I learned from this model
The first thing I understood is that ML models basically depend on mathematical structure.
If we use a different ML model, we are basically using a different mathematical structure. For example, we can have a Logistic Regression model, an HMM model, or different kinds of ML models. Underneath, everything is basically based on a mathematical structure.
Every machine-learning model uses a different mathematical structure and a different process to get to the point of making a prediction.

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

<img width="682" height="692" alt="image" src="https://github.com/user-attachments/assets/9a42b20d-f0c8-47ba-9275-f528a04b0dbb" />

from this information, i filter the data based on it datatype 

cleaning process:

<img width="1897" height="200" alt="image" src="https://github.com/user-attachments/assets/0cd0a347-9647-49be-bf07-424af69dc1f7" />

after that i replace the empty cell with zero value (we can also do interpolarization method to find the missing data, but i didn't do that in this exercise
coz i thought it may take sometime. so, i went to navie approach here)
---

## 3. Data Cleaning and Preparation

We removed:

```text
33 duplicate rows
```

leaving:

```text
7,010 customers
```

after that i seperate the training data and target data into two different dataframe

here, what i mean by traning data is the dataframe without chunk column in it and target data is chunk column only

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

<img width="602" height="151" alt="image" src="https://github.com/user-attachments/assets/38a0e673-6bda-49ca-9898-db87205f6937" />


The preprocessing is fitted only on the training data. Validation and test data are transformed using the already-fitted preprocessing steps.

This prevents information from the validation/test sets leaking into the training process.

---

## 5. Converting the Table into a Numerical Representation

now, i convert the 60 % of train data that seperate from dataset into numerical representation

like i taken numerical datatype object and used standard scaling. then used one-hotencoding on categroical data

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

<img width="525" height="235" alt="image" src="https://github.com/user-attachments/assets/50719d95-ffcf-4d6e-bec4-05cfec4f854b" />

---

## 7. Validation Results

<img width="505" height="475" alt="image" src="https://github.com/user-attachments/assets/b9c88ecd-b6b7-49a7-86e1-7f1bbf612792" />

Accuracy alone was not used as the main measure because the dataset is imbalanced: only about 26.5% of customers churn.


---

## 8. Project Structure

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

## 9. Visual Results

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
