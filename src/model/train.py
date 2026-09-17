"""
Model Training and Validation Module for Logistic Regression.

Orchestrates:
  1. Data Loading & Cleaning
  2. Stratified 3-way Split (Train / Val / Test)
  3. Leak-Free Preprocessing Pipeline (fit strictly on Train, transform Val & Test)
  4. Logistic Regression Optimization (minimizing Binary Cross-Entropy Loss)
  5. Validation Probability & Metric Inspection (Test set held out untouched)
  6. Model & Artifact Persistence
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE
)
from src.data.loader import load_data
from src.data.cleaner import clean_data
from src.data.splitter import split_data
from src.preprocessing.pipeline import create_preprocessing_pipeline
from src.model.logistic_regression import create_logistic_regression


def train_and_validate(
    data_path: Path = RAW_DATA_FILE,
    c_param: float = 1.0,
    class_weight: str | None = None,
    save_artifacts: bool = True
) -> dict:
    """
    Executes the full training and validation workflow.

    Parameters:
    -----------
    data_path : Path
        Path to raw CSV dataset.
    c_param : float, default=1.0
        Inverse regularization strength for Logistic Regression.
    class_weight : str | None, default=None
        Class weight setting (e.g. None or 'balanced').
    save_artifacts : bool, default=True
        Whether to save trained model & preprocessor to disk.

    Returns:
    --------
    dict
        Dictionary containing trained model, preprocessor, datasets, and validation metrics.
    """
    print("=" * 80)
    print("LOGISTIC REGRESSION TRAINING & VALIDATION PIPELINE")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. LOAD & CLEAN DATA
    # --------------------------------------------------------------------------
    print(f"\n[1] Loading dataset from: {data_path}")
    raw_df = load_data(str(data_path))

    print("[2] Cleaning dataset and extracting features X and target y...")
    X, y = clean_data(raw_df)
    print(f"    - Cleaned Total Samples: {len(X)} rows")
    print(f"    - Total Features: {X.shape[1]} columns")
    print(f"    - Overall Churn Rate: {y.mean() * 100:.2f}% ({y.sum()} churn / {len(y) - y.sum()} non-churn)")

    # --------------------------------------------------------------------------
    # 2. STRATIFIED 3-WAY SPLIT (TRAIN / VAL / TEST)
    # --------------------------------------------------------------------------
    print("\n[3] Splitting dataset into Stratified Train / Validation / Test partitions:")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=True
    )

    split_summary = [
        ["Train", len(X_train), f"{len(X_train) / len(X) * 100:.1f}%", y_train.sum(), f"{y_train.mean() * 100:.2f}%"],
        ["Validation", len(X_val), f"{len(X_val) / len(X) * 100:.1f}%", y_val.sum(), f"{y_val.mean() * 100:.2f}%"],
        ["Test (Held Out)", len(X_test), f"{len(X_test) / len(X) * 100:.1f}%", y_test.sum(), f"{y_test.mean() * 100:.2f}%"]
    ]
    print(tabulate(
        split_summary,
        headers=["Split Partition", "Samples", "Proportion", "Churn Count", "Churn %"],
        tablefmt="grid"
    ))

    # --------------------------------------------------------------------------
    # 3. LEAK-FREE PREPROCESSING (Fit on Train ONLY)
    # --------------------------------------------------------------------------
    print("\n[4] Fitting Preprocessing Pipeline STRICTLY on X_train...")
    preprocessor = create_preprocessing_pipeline()
    
    # Fit & transform training set
    X_train_proc = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    # Transform validation & test sets (NO fitting to prevent data leakage)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    print(f"    - Transformed feature dimensions: {X_train_proc.shape[1]} columns")
    print(f"    - X_train_processed shape: {X_train_proc.shape}")
    print(f"    - X_val_processed shape:   {X_val_proc.shape}")
    print(f"    - X_test_processed shape:  {X_test_proc.shape} (Stored & Held Out)")

    # --------------------------------------------------------------------------
    # 4. TRAIN LOGISTIC REGRESSION MODEL
    # --------------------------------------------------------------------------
    print(f"\n[5] Training Logistic Regression (C={c_param}, solver='lbfgs', class_weight={class_weight})...")
    model = create_logistic_regression(
        C=c_param,
        solver='lbfgs',
        class_weight=class_weight,
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    # Optimization: Find w and b that minimize Binary Cross-Entropy Loss
    model.fit(X_train_proc, y_train)
    print("    -> Optimization converged successfully.")

    # --------------------------------------------------------------------------
    # 5. MODEL PARAMETERS INSPECTION (Weights w and Bias b)
    # --------------------------------------------------------------------------
    intercept = model.intercept_[0]
    coefficients = model.coef_[0]

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient (Weight w)': coefficients,
        'Absolute Impact': np.abs(coefficients)
    }).sort_values(by='Coefficient (Weight w)', ascending=False)

    print(f"\n[6] Learned Model Parameters:")
    print(f"    - Bias / Intercept (b): {intercept:.4f}")
    
    top_pos = coef_df.head(5).values.tolist()
    top_neg = coef_df.tail(5).values.tolist()

    print("\n  Top 5 Features INCREASING Churn Probability (Positive Weights w > 0):")
    print(tabulate(top_pos, headers=['Feature', 'Weight (w)', '|Weight|'], tablefmt='simple', floatfmt='.4f'))

    print("\n  Top 5 Features DECREASING Churn Probability (Negative Weights w < 0):")
    print(tabulate(top_neg, headers=['Feature', 'Weight (w)', '|Weight|'], tablefmt='simple', floatfmt='.4f'))

    # --------------------------------------------------------------------------
    # 6. VALIDATION PROBABILITY & METRIC INSPECTION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION (Threshold = 0.50)")
    print("=" * 80)

    # Predict continuous probabilities p = sigma(Xw + b)
    val_probs = model.predict_proba(X_val_proc)[:, 1]
    val_preds_default = (val_probs >= 0.50).astype(int)

    # Probability Distribution Stats
    prob_percentiles = np.percentile(val_probs, [0, 25, 50, 75, 100])
    print(f"\nPredicted Probability Distribution p = P(Churn=1|x) on Validation Set:")
    print(f"  Min: {prob_percentiles[0]:.4f} | 25%: {prob_percentiles[1]:.4f} | Median: {prob_percentiles[2]:.4f} | 75%: {prob_percentiles[3]:.4f} | Max: {prob_percentiles[4]:.4f}")

    # Metrics computation
    val_acc = accuracy_score(y_val, val_preds_default)
    val_prec = precision_score(y_val, val_preds_default, zero_division=0)
    val_rec = recall_score(y_val, val_preds_default, zero_division=0)
    val_f1 = f1_score(y_val, val_preds_default, zero_division=0)
    val_roc_auc = roc_auc_score(y_val, val_probs)
    val_pr_auc = average_precision_score(y_val, val_probs)

    metrics_table = [
        ["ROC-AUC (Discrimination)", f"{val_roc_auc:.4f}"],
        ["PR-AUC (Avg Precision)", f"{val_pr_auc:.4f}"],
        ["Accuracy", f"{val_acc:.4f}"],
        ["Precision (at 0.50)", f"{val_prec:.4f}"],
        ["Recall (at 0.50)", f"{val_rec:.4f}"],
        ["F1-Score (at 0.50)", f"{val_f1:.4f}"],
    ]
    print("\n📊 VALIDATION METRICS SUMMARY")
    print("-" * 50)
    print(tabulate(metrics_table, headers=["Metric", "Validation Score"], tablefmt="grid"))

    # Confusion Matrix
    cm = confusion_matrix(y_val, val_preds_default)
    tn, fp, fn, tp = cm.ravel()
    cm_display = [
        ["Actual No Churn (0)", f"TN = {tn}", f"FP = {fp}"],
        ["Actual Churn (1)", f"FN = {fn}", f"TP = {tp}"]
    ]
    print("\n📊 CONFUSION MATRIX (Validation Set @ Threshold 0.50)")
    print("-" * 50)
    print(tabulate(cm_display, headers=["", "Pred No Churn (0)", "Pred Churn (1)"], tablefmt="grid"))

    # --------------------------------------------------------------------------
    # 7. SAVE ARTIFACTS
    # --------------------------------------------------------------------------
    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "logistic_regression.pkl"
        preprocessor_path = MODELS_DIR / "preprocessor.pkl"

        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        print(f"\n[7] Saved trained model to:        {model_path}")
        print(f"    Saved fitted preprocessor to: {preprocessor_path}")

    print("=" * 80)
    print("TRAINING & VALIDATION COMPLETE.")
    print("=" * 80)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "coef_df": coef_df,
        "val_probs": val_probs,
        "y_val": y_val,
        "metrics": {
            "accuracy": val_acc,
            "precision": val_prec,
            "recall": val_rec,
            "f1": val_f1,
            "roc_auc": val_roc_auc,
            "pr_auc": val_pr_auc
        },
        "datasets": {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "X_train_proc": X_train_proc, "X_val_proc": X_val_proc, "X_test_proc": X_test_proc
        }
    }


if __name__ == "__main__":
    results = train_and_validate()
