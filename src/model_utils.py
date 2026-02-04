"""
model_utils.py

Reusable modelling utilities for the COVID-19 intubation prediction project.

Scope (CV/GitHub-friendly)
-------------------------
This module centralises the modelling logic used across the notebooks:
- preprocessing (imputation + one-hot encoding)
- XGBoost model construction (with class-imbalance handling)
- optional SMOTE inside the CV pipeline (to avoid leakage)
- evaluation helpers (CV metrics + threshold sweep)

Design principles
-----------------
- Notebooks remain the primary narrative / analysis artefact.
- Functions here are small, explicit, and easy to audit.
- Defaults match the project notebooks (median/mode imputation, OHE with unknown handling).

Dependencies
------------
- scikit-learn
- xgboost
- imbalanced-learn (optional; only required if use_smote=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------
# Feature typing + preprocessing
# ---------------------------------------------------------------------
def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer numeric and categorical feature lists from dtypes.

    Parameters
    ----------
    X : pd.DataFrame

    Returns
    -------
    (numeric_features, categorical_features)
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(
    X: pd.DataFrame,
    numeric_features: Optional[Sequence[str]] = None,
    categorical_features: Optional[Sequence[str]] = None,
    ohe_dense: bool = True,
) -> ColumnTransformer:
    """
    Build a preprocessing transformer:
    - numeric: median imputation
    - categorical: most-frequent imputation + one-hot encoding

    Parameters
    ----------
    X : pd.DataFrame
        Training features (used to infer dtypes if feature lists not supplied).
    numeric_features : Optional[Sequence[str]]
        Explicit numeric columns (optional).
    categorical_features : Optional[Sequence[str]]
        Explicit categorical columns (optional).
    ohe_dense : bool
        If True, OneHotEncoder returns dense output (matches notebook behaviour).

    Returns
    -------
    ColumnTransformer
    """
    if numeric_features is None or categorical_features is None:
        num, cat = infer_feature_types(X)
        numeric_features = num if numeric_features is None else list(numeric_features)
        categorical_features = cat if categorical_features is None else list(categorical_features)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not ohe_dense)),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ],
        remainder="drop",
    )
    return preprocess


# ---------------------------------------------------------------------
# XGBoost model + pipeline
# ---------------------------------------------------------------------
def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute the standard XGBoost imbalance weight: neg/pos.

    Returns 1.0 if pos==0 to avoid division errors.
    """
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg / max(pos, 1))


def build_xgb_classifier(
    params: Optional[Mapping[str, Any]] = None,
    y: Optional[pd.Series] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> "Any":
    """
    Build an XGBClassifier with sensible defaults for this project.

    Parameters
    ----------
    params : Optional[Mapping[str, Any]]
        XGBoost hyperparameters to override defaults (e.g., from Optuna).
    y : Optional[pd.Series]
        If provided, scale_pos_weight is computed and injected unless explicitly set.
    random_state : int
    n_jobs : int

    Returns
    -------
    xgboost.XGBClassifier
    """
    from xgboost import XGBClassifier  # local import to keep module import light

    defaults: Dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }

    merged = dict(defaults)
    if params is not None:
        merged.update(dict(params))

    # Handle imbalance by default (unless already specified)
    if y is not None and "scale_pos_weight" not in merged:
        merged["scale_pos_weight"] = compute_scale_pos_weight(y)

    return XGBClassifier(**merged)


def build_pipeline(
    preprocess: ColumnTransformer,
    model: Any,
    use_smote: bool = False,
    random_state: int = 42,
) -> Any:
    """
    Build a modelling pipeline.

    If use_smote=True, uses an imbalanced-learn pipeline with SMOTE inserted
    *inside* the CV loop (preprocess -> SMOTE -> model), which prevents leakage.

    Returns a Pipeline-like object.
    """
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError as e:
            raise ImportError(
                "use_smote=True requires 'imbalanced-learn'. Install with: pip install imbalanced-learn"
            ) from e

        return ImbPipeline(steps=[
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=random_state)),
            ("model", model),
        ])

    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])


# ---------------------------------------------------------------------
# Cross-validation utilities
# ---------------------------------------------------------------------
def make_default_cv(
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> StratifiedKFold:
    """Create a default StratifiedKFold splitter."""
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def make_default_scoring() -> Dict[str, Any]:
    """
    Scorers consistent with the project notebooks.

    Returns
    -------
    dict
        Keys: f1, recall, precision
    """
    return {
        "f1": make_scorer(f1_score),
        "recall": make_scorer(recall_score),
        "precision": make_scorer(precision_score),
    }


def cross_validate_pipeline(
    pipe: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    scoring: Optional[Mapping[str, Any]] = None,
    n_jobs: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Run cross-validation for a pipeline and return raw fold scores.

    Notes
    -----
    - n_jobs defaults to 1 to avoid nested parallelism when tuning (Optuna + CV).
    """
    if cv is None:
        cv = make_default_cv()
    if scoring is None:
        scoring = make_default_scoring()

    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=dict(scoring),
        n_jobs=n_jobs,
        return_train_score=False,
    )
    return scores


# ---------------------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    f1: float
    recall: float
    precision: float


def threshold_sweep(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_proba: Sequence[float] | np.ndarray | pd.Series,
    thresholds: Iterable[float] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55),
) -> pd.DataFrame:
    """
    Compute precision/recall/F1 across a set of probability thresholds.

    Returns
    -------
    pd.DataFrame with columns: threshold, f1, recall, precision
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)

    rows: List[Dict[str, float]] = []
    for t in thresholds:
        y_pred = (y_proba_arr >= float(t)).astype(int)
        rows.append({
            "threshold": float(t),
            "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        })

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
