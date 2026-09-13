from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def aggregate_sparse_counts(
    x,
    obs: pd.DataFrame,
    var_names,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate cell-level counts into pseudobulk samples without densifying cell x gene data."""
    missing = [c for c in group_columns if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing grouping columns: {missing}")

    keys = obs[group_columns].astype(str).agg("||".join, axis=1)
    codes, levels = pd.factorize(keys, sort=True)
    n_groups = len(levels)
    n_cells = len(codes)
    selector = sparse.csr_matrix(
        (np.ones(n_cells, dtype=np.int8), (codes, np.arange(n_cells))),
        shape=(n_groups, n_cells),
    )
    x = sparse.csr_matrix(x)
    summed = selector @ x

    meta = pd.DataFrame([level.split("||") for level in levels], columns=group_columns)
    meta.index = pd.Index([f"pb_{i:03d}" for i in range(n_groups)], name="pseudobulk_id")
    counts = pd.DataFrame.sparse.from_spmatrix(summed, index=meta.index, columns=var_names)
    return counts, meta


def filter_pseudobulks_by_cell_number(
    obs: pd.DataFrame,
    group_columns: list[str],
    minimum: int,
) -> pd.DataFrame:
    sizes = obs.groupby(group_columns, observed=True).size().rename("n_nuclei").reset_index()
    return sizes.loc[sizes["n_nuclei"] >= minimum].copy()
