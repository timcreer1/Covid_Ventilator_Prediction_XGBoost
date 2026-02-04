"""
eval_utils.py

Evaluation utilities for the COVID-19 intubation prediction project.

Purpose (CV/GitHub-friendly)
----------------------------
- Produce consistent metric tables across experiments
- Support threshold analysis and probability-based curves (ROC/PR)
- Offer simple, reusable plotting helpers that save figures deterministically

Design principles
-----------------
- Keep notebooks readable; move reusable evaluation logic into src utilities
- Be explicit about what is computed (no hidden state)
- Avoid over-engineering; small, composable functions

Dependencies
------------
- numpy, pandas
- matplotlib
- scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------
def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_proba: Sequence[float] | np.ndarray | pd.Series,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Compute core binary classification metrics from probabilities at a given threshold.

    Returns
    -------
    dict with keys: f1, recall, precision, pr_auc, roc_auc
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    y_pred = (y_proba_arr >= float(threshold)).astype(int)

    out: Dict[str, float] = {}
    out["threshold"] = float(threshold)
    out["f1"] = float(f1_score(y_true_arr, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true_arr, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true_arr, y_pred, zero_division=0))

    # Probability-based metrics
    # ROC-AUC is undefined if y_true has only one class (guard for completeness)
    try:
        out["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba_arr))
    except ValueError:
        out["roc_auc"] = float("nan")

    try:
        out["pr_auc"] = float(average_precision_score(y_true_arr, y_proba_arr))
    except ValueError:
        out["pr_auc"] = float("nan")

    return out


def make_metrics_row(
    name: str,
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_proba: Sequence[float] | np.ndarray | pd.Series,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Create a single row for a results table.
    """
    m = compute_classification_metrics(y_true, y_proba, threshold=threshold)
    m["model"] = name
    return m


def metrics_table(
    rows: Sequence[Mapping[str, Any]],
    sort_by: str = "f1",
    descending: bool = True,
    round_dp: int = 3,
) -> pd.DataFrame:
    """
    Convert metric rows into a tidy DataFrame (and optionally sort + round).

    Expected columns (typical):
    - model, threshold, f1, recall, precision, pr_auc, roc_auc
    """
    df = pd.DataFrame(list(rows))

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)

    # Consistent rounding for display
    num_cols = [c for c in df.columns if c not in ("model",)]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").round(round_dp)

    # Reorder columns if present
    ordered = ["model", "threshold", "f1", "recall", "precision", "pr_auc", "roc_auc"]
    cols = [c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]
    return df[cols].reset_index(drop=True)


# ---------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------
def threshold_sweep(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_proba: Sequence[float] | np.ndarray | pd.Series,
    thresholds: Iterable[float] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55),
    round_dp: int = 3,
) -> pd.DataFrame:
    """
    Compute metrics at multiple thresholds.

    Returns
    -------
    DataFrame with columns: threshold, f1, recall, precision, pr_auc, roc_auc
    """
    rows = []
    for t in thresholds:
        rows.append(compute_classification_metrics(y_true, y_proba, threshold=float(t)))
    df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    # PR-AUC and ROC-AUC do not vary with threshold (kept for completeness)
    df = df.round(round_dp)
    return df


def best_threshold_by_metric(
    sweep_df: pd.DataFrame,
    metric: str = "f1",
) -> float:
    """
    Select threshold that maximises a chosen metric from a sweep table.
    """
    if metric not in sweep_df.columns:
        raise KeyError(f"Metric '{metric}' not in sweep_df columns: {list(sweep_df.columns)}")
    idx = sweep_df[metric].astype(float).idxmax()
    return float(sweep_df.loc[idx, "threshold"])


# ---------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------
def confusion_from_proba(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_proba: Sequence[float] | np.ndarray | pd.Series,
    threshold: float = 0.50,
) -> np.ndarray:
    """Compute confusion matrix from probabilities at a threshold."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba).astype(float) >= float(threshold)).astype(int)
    return confusion_matrix(y_true_arr, y_pred)


# ---------------------------------------------------------------------
# Curves + plotting helpers
# ---------------------------------------------------------------------
def roc_pr_curves(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_proba: Sequence[float] | np.ndarray | pd.Series,
) -> Dict[str, Any]:
    """
    Compute ROC and PR curve points.

    Returns
    -------
    dict with keys: roc_fpr, roc_tpr, roc_thresholds, pr_precision, pr_recall, pr_thresholds
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)

    fpr, tpr, roc_thr = roc_curve(y_true_arr, y_proba_arr)
    prec, rec, pr_thr = precision_recall_curve(y_true_arr, y_proba_arr)

    return {
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "roc_thresholds": roc_thr,
        "pr_precision": prec,
        "pr_recall": rec,
        "pr_thresholds": pr_thr,
    }


def savefig(fig_dir: Path | str, filename: str, dpi: int = 200) -> Path:
    """
    Save current matplotlib figure to fig_dir/filename.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / filename
    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    return out


def plot_roc_curve(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_proba: Sequence[float] | np.ndarray | pd.Series,
    title: str = "ROC Curve",
    fig_dir: Optional[Path | str] = None,
    filename: str = "roc_curve.png",
) -> None:
    """
    Plot ROC curve and optionally save.
    """
    curves = roc_pr_curves(y_true, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(curves["roc_fpr"], curves["roc_tpr"])
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)

    if fig_dir is not None:
        savefig(fig_dir, filename)


def plot_pr_curve(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_proba: Sequence[float] | np.ndarray | pd.Series,
    title: str = "Precision–Recall Curve",
    fig_dir: Optional[Path | str] = None,
    filename: str = "pr_curve.png",
) -> None:
    """
    Plot Precision–Recall curve and optionally save.
    """
    curves = roc_pr_curves(y_true, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(curves["pr_recall"], curves["pr_precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)

    if fig_dir is not None:
        savefig(fig_dir, filename)


# ---------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------
def save_table_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """
    Save a pandas DataFrame to CSV with parent directory creation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
