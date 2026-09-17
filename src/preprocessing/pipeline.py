"""
Preprocessing Pipeline for Telco Customer Churn Dataset.

Transforms raw/cleaned tabular features into standard mathematical matrices 
ready for Logistic Regression:
  - Continuous numerical features -> StandardScaler
  - Binary categorical features -> OneHotEncoder(drop='if_binary')
  - Binary numeric features (e.g. SeniorCitizen) -> Passthrough (already 0/1)
  - Multi-category features -> OneHotEncoder(drop='first')
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.data.loader import load_data
    from src.data.cleaner import clean_data
except ImportError:
    # Fallback when running inside subfolder
    sys.path.append(str(PROJECT_ROOT / "src"))
    from data.loader import load_data
    from data.cleaner import clean_data


# ==============================================================================
# 1. FEATURE GROUP DEFINITIONS
# ==============================================================================
# Continuous features requiring zero-mean, unit-variance scaling: (x - mu) / sigma
NUMERIC_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

# 2-category object features to be mapped to single 0/1 indicator columns
BINARY_CATEGORICAL_FEATURES = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling'
]

# Already binary (0/1) integer columns that require no scaling or encoding
PASSTHROUGH_FEATURES = [
    'SeniorCitizen'
]

# Multi-class categorical features to be dummy-encoded with drop='first' 
# to eliminate collinearity (dummy variable trap) in linear models
MULTI_CATEGORICAL_FEATURES = [
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod'
]


# ==============================================================================
# 2. PIPELINE FACTORY
# ==============================================================================
def create_preprocessing_pipeline(
    numeric_cols: list[str] = NUMERIC_FEATURES,
    binary_cols: list[str] = BINARY_CATEGORICAL_FEATURES,
    passthrough_cols: list[str] = PASSTHROUGH_FEATURES,
    multi_cat_cols: list[str] = MULTI_CATEGORICAL_FEATURES,
    drop_first: bool = True
) -> ColumnTransformer:
    """
    Constructs and returns an un-fitted scikit-learn ColumnTransformer.

    Parameters:
    -----------
    numeric_cols : list[str]
        Continuous numerical columns to scale with StandardScaler.
    binary_cols : list[str]
        Binary categorical columns to encode to 0/1 via OneHotEncoder(drop='if_binary').
    passthrough_cols : list[str]
        Columns already in 0/1 numeric format to pass through untransformed.
    multi_cat_cols : list[str]
        Categorical columns with >= 3 categories to encode via OneHotEncoder.
    drop_first : bool, default=True
        Whether to drop the first category in multi-class OHE to prevent multicollinearity.

    Returns:
    --------
    ColumnTransformer
        Configured preprocessor ready for fit() and transform().
    """
    numeric_transformer = StandardScaler()

    binary_transformer = OneHotEncoder(
        drop='if_binary',
        sparse_output=False,
        handle_unknown='ignore'
    )

    multi_cat_transformer = OneHotEncoder(
        drop='first' if drop_first else None,
        sparse_output=False,
        handle_unknown='ignore'
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('bin', binary_transformer, binary_cols),
            ('pass', 'passthrough', passthrough_cols),
            ('cat', multi_cat_transformer, multi_cat_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor


# ==============================================================================
# 3. PIPELINE INSPECTION & VERIFICATION
# ==============================================================================
def inspect_pipeline_output(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Fits the preprocessor on X, transforms X, and prints a comprehensive inspection report.

    Parameters:
    -----------
    preprocessor : ColumnTransformer
        The preprocessing pipeline to inspect.
    X : pd.DataFrame
        The feature DataFrame (cleaned, without target).

    Returns:
    --------
    pd.DataFrame
        Transformed dataset as a pandas DataFrame with descriptive column headers.
    """
    print("=" * 80)
    print("PREPROCESSING PIPELINE INSPECTION REPORT")
    print("=" * 80)
    
    print(f"\n[1] Input Feature Dimensions: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"    - Numerical features ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
    print(f"    - Binary categorical ({len(BINARY_CATEGORICAL_FEATURES)}): {BINARY_CATEGORICAL_FEATURES}")
    print(f"    - Binary passthrough ({len(PASSTHROUGH_FEATURES)}): {PASSTHROUGH_FEATURES}")
    print(f"    - Multi-class categorical ({len(MULTI_CATEGORICAL_FEATURES)}): {MULTI_CATEGORICAL_FEATURES}")

    # Fit and transform
    X_transformed_arr = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed_arr,
        columns=feature_names,
        index=X.index
    )

    print(f"\n[2] Transformed Feature Dimensions: {X_transformed_df.shape[0]} rows x {X_transformed_df.shape[1]} columns")
    
    # Verification checks
    has_nan = X_transformed_df.isnull().any().any()
    nan_count = X_transformed_df.isnull().sum().sum()
    all_numeric = all(np.issubdtype(dtype, np.number) for dtype in X_transformed_df.dtypes)

    print("\n[3] Mathematical & Integrity Validation:")
    print(f"    - Contains NaN / missing values: {has_nan} ({nan_count} NaNs)")
    print(f"    - All columns strictly numeric: {all_numeric}")
    print(f"    - Matrix shape: {X_transformed_df.shape}")

    # Feature table
    feature_summary = []
    for col in feature_names:
        series = X_transformed_df[col]
        feature_summary.append([
            col,
            str(series.dtype),
            round(float(series.min()), 3),
            round(float(series.mean()), 3),
            round(float(series.max()), 3),
            round(float(series.std()), 3),
            series.nunique()
        ])

    print("\n📊 TRANSFORMED COLUMNS SUMMARY TABLE")
    print("-" * 80)
    print(tabulate(
        feature_summary,
        headers=['Transformed Feature', 'Dtype', 'Min', 'Mean', 'Max', 'Std', 'Unique'],
        tablefmt='grid',
        stralign='left'
    ))

    print("\nFirst 3 rows of processed matrix (sample):")
    print(X_transformed_df.iloc[:3, :8].to_string())
    print("=" * 80)

    return X_transformed_df


if __name__ == '__main__':
    data_path = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    print(f"Loading raw data from: {data_path}")
    raw_df = load_data(str(data_path))

    print("Cleaning dataset...")
    X, y = clean_data(raw_df)

    print("Building preprocessing pipeline...")
    pipeline = create_preprocessing_pipeline()

    print("Inspecting pipeline transformation...")
    X_processed = inspect_pipeline_output(pipeline, X)
