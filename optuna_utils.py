"""
optuna_utils.py

Optuna utilities for the COVID-19 intubation prediction project.

Purpose (CV/GitHub-friendly)
----------------------------
- Keep Optuna tuning code clean and consistent across notebooks
- Provide a transparent, reproducible objective function for CV-based optimisation
- Offer lightweight progress logging suitable for notebooks (VS Code, Jupyter, Kaggle)

This module is intentionally "research-oriented":
- explicit search spaces
- simple study configuration
- no hidden magic

Expected usage (typical)
------------------------
1) Build a preprocessor and baseline model/pipeline (see model_utils.py)
2) Create an objective function with `make_xgb_objective(...)`
3) Create a study with `create_study(...)`
4) Run `study.optimize(objective, ...)` with `callbacks=[progress_callback(...)]`

Dependencies
------------
- optuna
- scikit-learn
- xgboost
- imbalanced-learn (only if use_smote=True)

See also
--------
- model_utils.py: preprocessing + pipeline builders
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import optuna

from sklearn.model_selection import StratifiedKFold

# Local project imports (keep import paths simple for a CV repo)
# If you prefer absolute imports, change to: from src.model_utils import ...
from model_utils import (
    build_pipeline,
    build_preprocessor,
    build_xgb_classifier,
    cross_validate_pipeline,
    make_default_cv,
    make_default_scoring,
)


# ---------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class XGBSearchSpace:
    """
    Search space bounds for XGBoost hyperparameters.

    These ranges are designed to be:
    - broad enough to explore meaningful model variation
    - bounded enough to run in reasonable time for a notebook experiment
    """
    n_estimators: Tuple[int, int] = (200, 900)
    max_depth: Tuple[int, int] = (3, 8)
    learning_rate: Tuple[float, float] = (0.01, 0.20)
    subsample: Tuple[float, float] = (0.60, 1.00)
    colsample_bytree: Tuple[float, float] = (0.60, 1.00)
    min_child_weight: Tuple[float, float] = (0.5, 10.0)
    gamma: Tuple[float, float] = (0.0, 5.0)
    reg_lambda: Tuple[float, float] = (0.0, 5.0)


def suggest_xgb_params(
    trial: optuna.trial.Trial,
    space: XGBSearchSpace = XGBSearchSpace(),
) -> Dict[str, Any]:
    """
    Suggest XGBoost hyperparameters for a trial.

    Returns
    -------
    dict of XGBClassifier params
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", *space.n_estimators),
        "max_depth": trial.suggest_int("max_depth", *space.max_depth),
        "learning_rate": trial.suggest_float("learning_rate", *space.learning_rate, log=True),
        "subsample": trial.suggest_float("subsample", *space.subsample),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *space.colsample_bytree),
        "min_child_weight": trial.suggest_float("min_child_weight", *space.min_child_weight),
        "gamma": trial.suggest_float("gamma", *space.gamma),
        "reg_lambda": trial.suggest_float("reg_lambda", *space.reg_lambda),
    }


# ---------------------------------------------------------------------
# Objective function builder
# ---------------------------------------------------------------------
def make_xgb_objective(
    X: pd.DataFrame,
    y: pd.Series,
    use_smote: bool,
    seed: int = 42,
    cv: Optional[StratifiedKFold] = None,
    scoring_key: str = "f1",
    n_jobs_cv: int = 1,
    space: XGBSearchSpace = XGBSearchSpace(),
    fixed_model_params: Optional[Mapping[str, Any]] = None,
) -> Callable[[optuna.trial.Trial], float]:
    """
    Create a CV-based Optuna objective for XGBoost.

    Parameters
    ----------
    X, y :
        Features and target.
    use_smote : bool
        If True, SMOTE is included inside the pipeline (preprocess -> SMOTE -> model).
    seed : int
        Reproducibility seed.
    cv : Optional[StratifiedKFold]
        CV splitter. If None, uses a standard StratifiedKFold.
    scoring_key : str
        Which scoring key to optimise (default: 'f1').
    n_jobs_cv : int
        Jobs used inside cross_validate. Use 1 when running Optuna to avoid nested parallelism.
    space : XGBSearchSpace
        Hyperparameter bounds.
    fixed_model_params : Optional[Mapping[str, Any]]
        Any fixed XGB params to force across all trials (e.g., tree_method, max_bin, etc.).

    Returns
    -------
    objective(trial) -> float
        Mean CV score for the chosen metric key.
    """
    if cv is None:
        cv = make_default_cv(n_splits=5, shuffle=True, random_state=seed)

    scoring = make_default_scoring()
    if scoring_key not in scoring:
        raise ValueError(f"scoring_key '{scoring_key}' not in default scoring keys: {list(scoring.keys())}")

    # Preprocessor inferred from X (consistent across trials)
    preprocess = build_preprocessor(X)

    def objective(trial: optuna.trial.Trial) -> float:
        # Suggest params and merge with any fixed params
        params = suggest_xgb_params(trial, space=space)
        if fixed_model_params:
            params.update(dict(fixed_model_params))

        model = build_xgb_classifier(params=params, y=y, random_state=seed, n_jobs=-1)
        pipe = build_pipeline(preprocess=preprocess, model=model, use_smote=use_smote, random_state=seed)

        scores = cross_validate_pipeline(
            pipe=pipe,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs_cv,
        )

        # cross_validate returns keys like 'test_f1', 'test_recall', etc.
        key = f"test_{scoring_key}"
        mean_score = float(np.mean(scores[key]))

        # Optional: store extra info for inspection
        trial.set_user_attr("mean_score", mean_score)

        return mean_score

    return objective


# ---------------------------------------------------------------------
# Study creation + persistence helpers
# ---------------------------------------------------------------------
def create_study(
    study_name: str,
    direction: str = "maximize",
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = True,
) -> optuna.Study:
    """
    Create an Optuna study with sensible defaults.

    Parameters
    ----------
    study_name : str
    direction : str
        'maximize' or 'minimize'
    sampler : Optional[optuna.samplers.BaseSampler]
        e.g., TPESampler(seed=...)
    pruner : Optional[optuna.pruners.BasePruner]
        e.g., MedianPruner(n_warmup_steps=...)
    storage : Optional[str]
        If provided, a storage URL (e.g., "sqlite:///optuna_study.db") to persist trials.
        If None, study is in-memory.
    load_if_exists : bool
        If using persistent storage, reuse an existing study of same name.

    Returns
    -------
    optuna.Study
    """
    if sampler is None:
        sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    return study


def make_sqlite_storage(db_path: Path | str) -> str:
    """
    Convenience to build a sqlite storage URL for Optuna.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


# ---------------------------------------------------------------------
# Progress logging (callbacks)
# ---------------------------------------------------------------------
def progress_callback(total_trials: int) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """
    Optuna callback for per-trial progress logging.

    Works well in notebooks and provides a compact, informative line per trial.
    """
    import time

    t0 = time.time()

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        elapsed_min = (time.time() - t0) / 60
        best = study.best_value if study.best_trial is not None else None
        n_done = len(study.trials)

        val = trial.value if trial.value is not None else float("nan")
        best_str = f"{best:.4f}" if best is not None else "nan"

        print(
            f"[Optuna] Trial {n_done}/{total_trials} | "
            f"value={val:.4f} | best={best_str} | elapsed={elapsed_min:.1f} min"
        )

    return _callback


# ---------------------------------------------------------------------
# Utilities for summarising results
# ---------------------------------------------------------------------
def study_to_frame(study: optuna.Study) -> pd.DataFrame:
    """
    Convert an Optuna study to a tidy DataFrame of trials (including params).
    """
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    # Sort best-first for maximise, worst-first for minimise
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        df = df.sort_values("value", ascending=False)
    else:
        df = df.sort_values("value", ascending=True)
    return df.reset_index(drop=True)


def best_params(study: optuna.Study) -> Dict[str, Any]:
    """
    Return best trial params as a plain dict.
    """
    return dict(study.best_trial.params)


def save_study_csv(study: optuna.Study, path: Path | str) -> None:
    """
    Save a study trials table to CSV for reproducibility / reporting.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = study_to_frame(study)
    df.to_csv(path, index=False)
