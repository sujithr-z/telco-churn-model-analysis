"""
Project Configuration & Feature Definitions for Telco Customer Churn.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "report"

RAW_DATA_FILE = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Target & ID Columns
TARGET_COL = "Churn"
ID_COL = "customerID"

# ------------------------------------------------------------------------------
# Feature Groups for Preprocessing Pipeline
# ------------------------------------------------------------------------------

# 1. Continuous numeric features -> StandardScaler (mean=0, variance=1)
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# 2. Binary object features -> Encoded to 0/1 via OneHotEncoder(drop='if_binary')
BINARY_CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]

# 3. Binary numeric features -> Passthrough (already 0/1 integers)
PASSTHROUGH_FEATURES = [
    "SeniorCitizen"
]

# 4. Multi-class categorical features (>= 3 categories) -> OneHotEncoder(drop='first')
MULTI_CATEGORICAL_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

# All input features combined
ALL_FEATURE_COLS = (
    NUMERIC_FEATURES
    + BINARY_CATEGORICAL_FEATURES
    + PASSTHROUGH_FEATURES
    + MULTI_CATEGORICAL_FEATURES
)

# ------------------------------------------------------------------------------
# Experiment / Split Settings
# ------------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
