"""
Gradient Boosting Model Definition, Factory, and Hyperparameter Tuning.

Provides GradientBoostingClassifier instances and systematic hyperparameter tuning 
using Stratified K-Fold Cross-Validation on the training partition.
"""

from typing import Optional, Any
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def create_gradient_boosting(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    subsample: float = 0.8,
    random_state: int = 42
) -> GradientBoostingClassifier:
    """
    Creates and returns an un-fitted GradientBoostingClassifier.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting stages.
    learning_rate : float, default=0.1
        Step size shrinkage to prevent overfitting.
    max_depth : int, default=3
        Maximum depth of individual regression trees (controls interaction order).
    min_samples_split : int, default=20
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=10
        Minimum number of samples required to be at a leaf node.
    subsample : float, default=0.8
        Fraction of samples used for fitting individual base learners (Stochastic Gradient Boosting).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    GradientBoostingClassifier
        Configured gradient boosting estimator.
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state
    )


def tune_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[dict[str, list[Any]]] = None,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42
) -> tuple[GradientBoostingClassifier, dict, GridSearchCV]:
    """
    Performs systematic hyperparameter tuning using Stratified K-Fold Cross-Validation on X_train.

    Parameters:
    -----------
    X_train : np.ndarray
        Transformed training feature matrix.
    y_train : np.ndarray
        Training labels (0/1).
    param_grid : dict, optional
        Search grid for hyperparameters.
    cv_folds : int, default=5
        Number of cross-validation folds.
    scoring : str, default='roc_auc'
        Optimization metric for model selection.
    random_state : int, default=42
        Random seed for CV splitting.

    Returns:
    --------
    tuple:
        (best_model, best_params, grid_search_object)
    """
    if param_grid is None:
        param_grid = {
            'max_depth': [2, 3, 4, 5],
            'learning_rate': [0.02, 0.05, 0.1],
            'n_estimators': [100, 150, 200],
            'min_samples_leaf': [10, 20, 40],
            'subsample': [0.8, 1.0]
        }

    base_estimator = GradientBoostingClassifier(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search
