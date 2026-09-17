import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

# Configure UTF-8 output for Windows console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.loader import load_data
from src.data.cleaner import clean_data


def classify_column(series: pd.Series) -> str:
    if series.dtype in ['int64', 'float64']:
        return 'Numerical'
    elif series.dtype == 'object':
        unique_count = series.nunique()
        if unique_count == 2:
            return 'Binary Categorical'
        else:
            return 'Categorical'
    else:
        return 'Unknown'

def inspect_data(X:pd.DataFrame , y: pd.Series = None) -> None:
    print("=" * 80)
    print("DATASET INSPECTION REPORT")
    print("=" * 80)
    
    summary_data = []

    for col in X.columns:
        series = X[col]
        dtype = str(series.dtype)
        category = classify_column(series)
        unique_count = series.nunique()
        missing_count = series.isnull().sum()
        missing_pct = (missing_count / len(series))*100

        summary_data.append([
            col,
            dtype,
            category,
            unique_count,
            f"{missing_count} ({missing_pct:.2f}%)"
        ])

    # Add target if provided
    if y is not None:
        summary_data.append([
            'Churn (Target)',
            str(y.dtype),
            'Binary',
            y.nunique(),
            f"{y.isnull().sum()} ({(y.isnull().sum() / len(y)) * 100:.2f}%)"
        ])
    
    print("\n📊 COLUMN SUMMARY")
    print("-" * 80)
    print(tabulate(
        summary_data,
        headers=['Column', 'Data Type', 'Category', 'Unique Values', 'Missing'],
        tablefmt='grid',
        stralign='left'
    ))
    
    # Detailed breakdown for categorical and binary columns
    print("\n" + "=" * 80)
    print("DETAILED CATEGORICAL/BINARY BREAKDOWN")
    print("=" * 80)
    
    for col in X.columns:
        category = classify_column(X[col])
        
        if category in ['Binary', 'Categorical', 'Binary Categorical']:
            print(f"\n🔹 {col} ({category})")
            print("-" * 40)
            
            unique_values = X[col].dropna().unique()
            value_counts = X[col].value_counts()
            
            for val in sorted(unique_values):
                count = value_counts[val]
                pct = (count / len(X)) * 100
                print(f"  ├─ {val:20s} : {count:5d} ({pct:5.2f}%)")
    
    # Target distribution
    if y is not None:
        print(f"\n🔹 Churn (Target)")
        print("-" * 40)
        value_counts = y.value_counts()
        for val, count in value_counts.items():
            pct = (count / len(y)) * 100
            print(f"  ├─ {str(val):20s} : {count:5d} ({pct:5.2f}%)")

if __name__ == '__main__':
    data_path = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    raw_df = load_data(str(data_path))
    X, y = clean_data(raw_df)
    inspect_data(X, y)