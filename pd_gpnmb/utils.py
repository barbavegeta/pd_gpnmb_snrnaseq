from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def snake_case(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def canonical_sample_id(value: str) -> str:
    """Map variants such as s.0096, s_0096 and S0096 to s_0096."""
    value = str(value).strip()
    match = re.search(r"s[._-]?(\d{3,5})", value, flags=re.IGNORECASE)
    if not match:
        return value
    return f"s_{match.group(1)}"


def sample_from_cell_id(cell_id: str) -> str:
    match = re.match(r"^(s[._-]?\d{3,5})", str(cell_id), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot extract sample ID from cell ID: {cell_id}")
    return canonical_sample_id(match.group(1))


def canonical_cell_id(cell_id: str) -> str:
    """Normalise only the donor prefix while preserving the 10x barcode."""
    cell_id = str(cell_id).strip()
    match = re.match(r"^(s[._-]?\d{3,5})([_-].+)$", cell_id, flags=re.IGNORECASE)
    if not match:
        return cell_id
    return f"{canonical_sample_id(match.group(1))}{match.group(2)}"


def first_existing(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    columns = list(columns)
    lookup = {snake_case(c): c for c in columns}
    for alias in aliases:
        if snake_case(alias) in lookup:
            return lookup[snake_case(alias)]
    return None


def canonical_condition(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "parkinson" in text or text in {"pd", "case", "1"}:
        return "PD"
    if "control" in text or text in {"ctrl", "hc", "healthy", "0"}:
        return "Control"
    return str(value).strip()


def canonical_sex(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"m", "male", "man"}:
        return "Male"
    if text in {"f", "female", "woman"}:
        return "Female"
    return str(value).strip()


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg correction, preserving missing values."""
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    ok = pvalues.notna()
    if not ok.any():
        return result
    p = pvalues.loc[ok].astype(float).to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    result.loc[ok] = restored
    return result
