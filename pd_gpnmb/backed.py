from __future__ import annotations

import numpy as np
from scipy import sparse


def _row_sum(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.sum(axis=1)).ravel()
    return np.asarray(x).sum(axis=1)


def _row_nnz(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray((x > 0).sum(axis=1)).ravel()
    return (np.asarray(x) > 0).sum(axis=1)


def _normalise_row_mask(adata, row_mask):
    if row_mask is None:
        return np.ones(adata.n_obs, dtype=bool)
    row_mask = np.asarray(row_mask, dtype=bool)
    if row_mask.shape != (adata.n_obs,):
        raise ValueError(
            f"row_mask has shape {row_mask.shape}; expected ({adata.n_obs},)"
        )
    return row_mask


def streamed_qc(adata, mt_mask: np.ndarray, row_mask=None, chunk_size: int = 2048):
    """Compute basic QC from a backed AnnData without creating backed views.

    row_mask is applied chunk-by-chunk to the original backed AnnData, avoiding
    AnnData's unsupported view-of-a-view indexing behaviour.
    """
    row_mask = _normalise_row_mask(adata, row_mask)
    n_keep = int(row_mask.sum())
    total = np.zeros(n_keep, dtype=np.float64)
    n_genes = np.zeros(n_keep, dtype=np.int32)
    mt_counts = np.zeros(n_keep, dtype=np.float64)

    out_start = 0
    for x, start, end in adata.chunked_X(chunk_size):
        keep = row_mask[start:end]
        if not keep.any():
            continue
        xs = x[keep]
        k = int(keep.sum())
        out_end = out_start + k
        total[out_start:out_end] = _row_sum(xs)
        n_genes[out_start:out_end] = _row_nnz(xs)
        if mt_mask.any():
            mt_counts[out_start:out_end] = _row_sum(xs[:, mt_mask])
        out_start = out_end

    pct_mt = np.divide(
        mt_counts * 100.0,
        total,
        out=np.zeros_like(mt_counts),
        where=total > 0,
    )
    return total, n_genes, pct_mt


def small_gene_matrix(adata, genes: list[str], row_mask=None):
    """Load selected genes from the original backed AnnData into memory.

    The gene slice is taken exactly once from the backed root object. Any row
    filtering is then performed in memory, so no backed view is indexed again.
    """
    genes = [g for g in genes if g in adata.var_names]
    row_mask = _normalise_row_mask(adata, row_mask)
    n_keep = int(row_mask.sum())
    if not genes:
        return genes, np.empty((n_keep, 0), dtype=np.float32)

    sub = adata[:, genes].to_memory()
    x = sub.X[row_mask]
    if sparse.issparse(x):
        x = x.toarray()
    return genes, np.asarray(x, dtype=np.float32)
