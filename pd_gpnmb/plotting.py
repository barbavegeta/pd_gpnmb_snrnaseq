from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def savefig(path: str | Path, dpi: int = 180) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def donor_boxplot(df: pd.DataFrame, value: str, group: str, path: str | Path, ylabel: str) -> None:
    groups = [g for g in ["Control", "PD"] if g in set(df[group].dropna())]
    values = [df.loc[df[group] == g, value].dropna().to_numpy() for g in groups]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(values, tick_labels=groups, showfliers=False)
    for i, vals in enumerate(values, start=1):
        ax.scatter([i] * len(vals), vals, s=22, alpha=0.7)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    savefig(path)
