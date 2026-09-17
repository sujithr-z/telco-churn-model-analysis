"""
Logistic Regression Model Definition and Factory.

Provides standard and customized Logistic Regression classifiers for binary churn prediction.
Computes linear logit: z = Xw + b, and sigmoid probability: p = 1 / (1 + exp(-z)).
"""

from typing import Optional, Literal
from sklearn.linear_model import LogisticRegression


def create_logistic_regression(
    C: float = 1.0,
    penalty: Optional[str] = None,
    solver: str = 'lbfgs',
    class_weight: Optional[str | dict] = None,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    kwargs = {
        'C': C,
        'solver': solver,
        'class_weight': class_weight,
        'max_iter': max_iter,
        'random_state': random_state
    }
    if penalty is not None:
        kwargs['penalty'] = penalty

    return LogisticRegression(**kwargs)
