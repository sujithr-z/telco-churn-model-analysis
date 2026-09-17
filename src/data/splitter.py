"""
Dataset Splitting Module.

Performs stratified Train / Validation / Test splitting to ensure leak-free evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits features X and labels y into stratified Train, Validation, and Test partitions.

    Parameters:
    -----------
    X : pd.DataFrame
        Cleaned feature matrix.
    y : pd.Series
        Binary target series (0/1).
    val_size : float, default=0.15
        Proportion of dataset for validation partition (e.g. 0.15 = 15%).
    test_size : float, default=0.15
        Proportion of dataset for test partition (e.g. 0.15 = 15%).
    random_state : int, default=42
        Random seed for reproducible shuffling.
    stratify : bool, default=True
        Whether to preserve class proportions across all splits.

    Returns:
    --------
    tuple:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    stratify_col = y if stratify else None

    # Step 1: Hold out Test set (e.g. 15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    # Step 2: Split remaining data into Train and Validation
    # Relative validation size: val_size / (1.0 - test_size)
    relative_val_size = val_size / (1.0 - test_size)
    stratify_temp = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=stratify_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
