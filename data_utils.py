"""
data_utils.py

Utility functions for the COVID-19 intubation prediction project (Mexico clinical dataset, Kaggle:
"meirnizri/covid19-dataset").

Design goals:
- Keep notebooks readable (analysis + narrative)
- Centralise reusable, testable data prep logic
- Be explicit about dataset coding conventions (1/2/97/98/99, 9999-99-99)

Notes on dataset conventions
----------------------------
- Many binary clinical indicators: 1 = Yes, 2 = No
- Unknown / not specified commonly encoded as: 97 / 98 / 99
- DATE_DIED uses '9999-99-99' for patients recorded as alive

This module applies *minimal, transparent* recoding suitable for university research workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
UNKNOWN_CODES: set[int] = {97, 98, 99}
DEFAULT_LEAKAGE_COLS: tuple[str, ...] = ("icu", "date_died", "patient_type")
SOURCE_OF_TARGET_COLS: tuple[str, ...] = ("intubed",)  # raw field used to create y


# ---------------------------------------------------------------------
# File + loading helpers
# ---------------------------------------------------------------------
def resolve_data_path(
    data_dir: Path | str,
    candidates: Sequence[str] = ("Covid_Data.csv", "Covid Data.csv"),
) -> Path:
    """
    Resolve the raw CSV path from a directory by checking candidate filenames.

    Parameters
    ----------
    data_dir : Path | str
        Directory containing the raw CSV.
    candidates : Sequence[str]
        Candidate filenames to check.

    Returns
    -------
    Path
        First existing candidate path.

    Raises
    ------
    FileNotFoundError
        If none of the candidate filenames exist.
    """
    data_dir = Path(data_dir)
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these found in {data_dir}: {list(candidates)}")


def load_raw_csv(
    csv_path: Path | str,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Load the raw CSV as distributed (no recoding).

    Parameters
    ----------
    csv_path : Path | str
        Path to the raw CSV.
    low_memory : bool
        Passed to pandas.read_csv.

    Returns
    -------
    pd.DataFrame
    """
    csv_path = Path(csv_path)
    return pd.read_csv(csv_path, low_memory=low_memory)


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names to lower case and strip surrounding whitespace.

    IMPORTANT: We avoid aggressive snake_case transforms here because this dataset already uses
    uppercase + underscores (e.g., MEDICAL_UNIT). Lowercasing preserves structure safely.

    Returns a copy.
    """
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


# ---------------------------------------------------------------------
# Recoding helpers
# ---------------------------------------------------------------------
def coerce_numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric; invalid parsing -> NaN."""
    return pd.to_numeric(s, errors="coerce")


def recode_yes_no(s: pd.Series, unknown_codes: Iterable[int] = UNKNOWN_CODES) -> pd.Series:
    """
    Recode common yes/no coding: {1:1, 2:0, 97/98/99:NaN}.

    Parameters
    ----------
    s : pd.Series
        Input coded series.
    unknown_codes : Iterable[int]
        Values treated as missing.

    Returns
    -------
    pd.Series
        Float series with values in {0.0, 1.0, NaN}.
    """
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(~x.isin(list(unknown_codes)), np.nan)
    return x.map({1: 1, 2: 0})


def derive_died_from_date(df: pd.DataFrame, date_col: str = "date_died") -> pd.DataFrame:
    """
    Create a derived 'died' indicator from the 'date_died' column.

    - '9999-99-99' is treated as alive (died=0)
    - Otherwise died=1 (even if the exact date fails parsing)

    Adds:
      - died (int)
      - date_died_parsed (datetime64[ns], optional)

    Returns a copy.
    """
    out = df.copy()
    if date_col not in out.columns:
        return out

    s = out[date_col].astype(str)
    out["died"] = np.where(s.eq("9999-99-99") | s.eq("nan"), 0, 1).astype(int)
    out["date_died_parsed"] = pd.to_datetime(out[date_col], errors="coerce")
    return out


# ---------------------------------------------------------------------
# Leakage control
# ---------------------------------------------------------------------
def drop_columns_if_present(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Drop columns if present, returning a copy."""
    out = df.copy()
    existing = [c for c in cols if c in out.columns]
    if existing:
        out = out.drop(columns=existing)
    return out


def drop_leakage_features(X: pd.DataFrame, leakage_cols: Sequence[str] = DEFAULT_LEAKAGE_COLS) -> pd.DataFrame:
    """
    Remove downstream-care / post-outcome indicators that can leak information.

    Typical examples in this dataset:
    - icu
    - date_died
    - patient_type

    Returns a copy.
    """
    return drop_columns_if_present(X, list(leakage_cols))


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PrepConfig:
    """
    Configuration for preparing a modelling dataset.

    Parameters
    ----------
    target_source_col : str
        Raw coded column used to derive the modelling target.
    target_name : str
        Output target column name.
    leakage_cols : Sequence[str]
        Columns to exclude from predictors due to leakage concerns.
    binary_cols : Optional[Sequence[str]]
        If provided, these columns will be recoded with 1/2/97 scheme.
        If None, only the target is recoded (minimal approach).
    """
    target_source_col: str = "intubed"
    target_name: str = "intubated"
    leakage_cols: Sequence[str] = DEFAULT_LEAKAGE_COLS
    binary_cols: Optional[Sequence[str]] = None


DEFAULT_BINARY_LIKE: tuple[str, ...] = (
    "pneumonia", "pregnant", "diabetes", "copd", "asthma", "inmsupr",
    "hypertension", "other_disease", "cardiovascular", "obesity",
    "renal_chronic", "tobacco", "icu",
)


def prepare_modelling_frame(
    df_raw: pd.DataFrame,
    config: PrepConfig = PrepConfig(),
    drop_unknown_target: bool = True,
) -> pd.DataFrame:
    """
    Prepare a modelling-ready DataFrame with:
    - standardised columns (lowercase)
    - derived target column (intubated) from the raw coded source (intubed)
    - optional recoding of a set of binary columns
    - optional 'died' derived label (if date_died exists)
    - minimal numeric coercions (age)

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw dataframe loaded from CSV.
    config : PrepConfig
        Preparation configuration.
    drop_unknown_target : bool
        If True, removes rows where target is unknown (97/98/99).

    Returns
    -------
    pd.DataFrame
        Prepared dataframe including target column.
    """
    df = standardise_columns(df_raw)

    # Derive target
    if config.target_source_col not in df.columns:
        raise KeyError(f"Expected column '{config.target_source_col}' not found after standardisation.")

    df[config.target_name] = recode_yes_no(df[config.target_source_col])

    if drop_unknown_target:
        df = df.dropna(subset=[config.target_name]).copy()
        df[config.target_name] = df[config.target_name].astype(int)

    # Optional: recode a set of binary-like columns
    if config.binary_cols is None:
        binary_cols = [c for c in DEFAULT_BINARY_LIKE if c in df.columns]
    else:
        binary_cols = [c for c in config.binary_cols if c in df.columns]

    for c in binary_cols:
        df[c] = recode_yes_no(df[c])

    # Derived label from death date (kept as feature by default; leak control handled later)
    df = derive_died_from_date(df, date_col="date_died")

    # Minimal numeric coercions
    if "age" in df.columns:
        df["age"] = coerce_numeric(df["age"])

    return df


def split_X_y(
    df: pd.DataFrame,
    target_col: str = "intubated",
    drop_source_of_target: bool = True,
    source_cols: Sequence[str] = SOURCE_OF_TARGET_COLS,
    drop_leakage: bool = True,
    leakage_cols: Sequence[str] = DEFAULT_LEAKAGE_COLS,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create X and y from a prepared dataframe.

    This function optionally:
    - removes the source-of-target column(s) (e.g., 'intubed') from X
    - removes leakage columns from X

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataframe (must include target_col).
    target_col : str
        Name of target column.
    drop_source_of_target : bool
        If True, remove source_cols from X.
    source_cols : Sequence[str]
        Columns to remove because they directly encode the target.
    drop_leakage : bool
        If True, remove leakage_cols from X.
    leakage_cols : Sequence[str]
        Leakage columns to drop.

    Returns
    -------
    (X, y)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    if drop_source_of_target:
        X = drop_columns_if_present(X, list(source_cols))

    if drop_leakage:
        X = drop_leakage_features(X, leakage_cols=leakage_cols)

    return X, y


# ---------------------------------------------------------------------
# Simple reporting helpers (optional)
# ---------------------------------------------------------------------
def target_proportion(y: pd.Series) -> pd.Series:
    """Return class proportions for a binary target."""
    return y.value_counts(normalize=True).rename("proportion")


def missingness_percent(df: pd.DataFrame) -> pd.Series:
    """Percent missing per column, descending."""
    return (df.isna().mean() * 100).sort_values(ascending=False)


def unknown_code_percent(df: pd.DataFrame, unknown_codes: Iterable[int] = UNKNOWN_CODES) -> pd.Series:
    """
    Percent of values in each column that are encoded as unknown (97/98/99).
    Only meaningful for numeric-coded columns.
    """
    out: Dict[str, float] = {}
    unk = set(int(x) for x in unknown_codes)
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            out[c] = float(s.isin(list(unk)).mean() * 100)
    return pd.Series(out).sort_values(ascending=False)


# ---------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------
def save_parquet(df: pd.DataFrame, path: Path | str, **kwargs: Any) -> None:
    """Save dataframe to parquet (wrapper for consistency)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)


def load_parquet(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Load dataframe from parquet."""
    return pd.read_parquet(Path(path), **kwargs)
