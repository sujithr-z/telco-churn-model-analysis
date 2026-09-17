import pandas as pd
import numpy as np
try:
    from src.data.loader import load_data
except ImportError:
    from loader import load_data
from time import sleep

def clean_data(df : pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_clean = df.copy()

    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
    
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['TotalCharges'])
    df_clean = df_clean.drop_duplicates()
    if len(df_clean) < initial_rows:
        print(f"[Cleaner] Dropped {initial_rows - len(df_clean)} duplicate rows.")

    if 'tenure' in df_clean.columns:
        invalid_tenure = df_clean[df_clean['tenure'] < 0]
        if not invalid_tenure.empty:
            print(f"[Cleaner] Warning: Found {len(invalid_tenure)} rows with negative tenure. Dropping.")
            df_clean = df_clean[df_clean['tenure'] >= 0]

    if 'Churn' not in df_clean.columns:
        raise ValueError("Target column 'Churn' not found in dataset.")

    y = df_clean['Churn'].map({'Yes':1, 'No':0})
    if y.isnull().any():
        raise ValueError("Target column contains invalid values (must be 'Yes' or 'No').")
    X = df_clean.drop(columns=['Churn'])

    return X, y

if __name__ == "__main__":
    print("Loading data...")
    sleep(3)

    df = load_data("D:\\agentic AI\\MODELS\\ACM-SIG-AI-TELCO-CHURN\\data\\raw\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

    print("\n--- BEFORE CLEANING ---")
    print(f"TotalCharges dtype: {df['TotalCharges'].dtype}")
    print(f"Total rows: {len(df)}")

    X_TrainData, y_targetData = clean_data(df)

    print("\n--- AFTER CLEANING ---")
    print(f"TotalCharges dtype: {X_TrainData['TotalCharges'].dtype}")
    print(f"TotalCharges NaN count: {X_TrainData['TotalCharges'].isnull().sum()}")
    print(f"Total rows: {len(X_TrainData)}")
    print(f"Target (y) unique values: {y_targetData.unique()}")
    print(f"Features (X) columns: {list(X_TrainData.columns)}")
