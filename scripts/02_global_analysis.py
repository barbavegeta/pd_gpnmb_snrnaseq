#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pd_gpnmb.backed import streamed_qc, small_gene_matrix
from pd_gpnmb.utils import load_config, ensure_dirs


def marker_panel(cfg: dict, genes: pd.Index) -> list[str]:
    ordered: list[str] = []
    for marker_list in cfg["markers"].values():
        ordered.extend(marker_list)
    seen: set[str] = set()
    return [g for g in ordered if g in genes and not (g in seen or seen.add(g))]


def save_categorical_umap(obs: pd.DataFrame, path: Path) -> None:
    use = obs.dropna(subset=["author_umap_1", "author_umap_2", "author_cell_type"]).copy()
    fig, ax = plt.subplots(figsize=(9, 7))
    for label, df in use.groupby("author_cell_type", observed=True):
        ax.scatter(df["author_umap_1"], df["author_umap_2"], s=1.0, alpha=0.65,
                   rasterized=True, label=str(label))
    ax.set_xlabel("Author UMAP 1")
    ax.set_ylabel("Author UMAP 2")
    ax.set_title("GSE243639 author cell-type annotations")
    ax.legend(markerscale=5, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_target_umap(obs: pd.DataFrame, expr: np.ndarray, path: Path, target: str) -> None:
    keep = obs[["author_umap_1", "author_umap_2"]].notna().all(axis=1).to_numpy()
    coords = obs.loc[keep, ["author_umap_1", "author_umap_2"]].to_numpy(float)
    vals = expr[keep]
    fig, ax = plt.subplots(figsize=(8, 7))
    pts = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=1.1, rasterized=True)
    fig.colorbar(pts, ax=ax, label=f"log1p normalised {target}")
    ax.set_xlabel("Author UMAP 1")
    ax.set_ylabel("Author UMAP 2")
    ax.set_title(f"{target} expression")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_marker_dotplot(obs: pd.DataFrame, genes: list[str], x: np.ndarray,
                        total_counts: np.ndarray, path: Path) -> None:
    denom = np.maximum(total_counts.astype(np.float32), 1.0)[:, None]
    norm = np.log1p((x / denom) * 1e4)
    labels = obs["author_cell_type"].astype("string")
    valid = labels.notna().to_numpy()
    categories = sorted(labels[valid].unique().tolist())

    mean_rows = []
    frac_rows = []
    for label in categories:
        m = (labels == label).fillna(False).to_numpy()
        mean_rows.append(norm[m].mean(axis=0))
        frac_rows.append((x[m] > 0).mean(axis=0))
    means = np.vstack(mean_rows)
    fracs = np.vstack(frac_rows)

    fig_w = max(10, 0.32 * len(genes) + 4)
    fig_h = max(6, 0.45 * len(categories) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    xx, yy = np.meshgrid(np.arange(len(genes)), np.arange(len(categories)))
    sc = ax.scatter(xx.ravel(), yy.ravel(), c=means.ravel(),
                    s=15 + 180 * fracs.ravel(), rasterized=True)
    fig.colorbar(sc, ax=ax, label="Mean log1p normalised expression")
    ax.set_xticks(np.arange(len(genes)), genes, rotation=90)
    ax.set_yticks(np.arange(len(categories)), categories)
    ax.set_xlabel("Marker gene")
    ax.set_ylabel("Author cell type")
    ax.set_title("Marker validation (dot size = detection fraction)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    accession = cfg["project"]["accession"]
    processed = Path(cfg["paths"]["processed"])
    results = Path(cfg["paths"]["results"])
    figs = results / "figures"
    tables = results / "tables"
    ensure_dirs(figs, tables)

    inp = processed / f"{accession}_raw_qc_annotated.h5ad"
    print(f"Opening in backed mode: {inp}", flush=True)
    adata = ad.read_h5ad(inp, backed="r")

    if "author_cell_type" not in adata.obs:
        raise KeyError("author_cell_type is missing from the annotated H5AD")

    # Keep the root backed object. Do not create a backed AnnData view and then
    # slice it again later; AnnData deliberately forbids view-of-a-view indexing.
    labelled = adata.obs["author_cell_type"].notna().to_numpy()
    n_atlas = int(labelled.sum())
    if n_atlas == 0:
        raise ValueError("No nuclei have author cell-type labels")
    print(f"Author-QC atlas: {n_atlas:,} nuclei x {adata.n_vars:,} genes", flush=True)

    mt_mask = np.asarray(adata.var_names.str.upper().str.startswith("MT-"), dtype=bool)
    print("Computing QC metrics in disk-backed chunks...", flush=True)
    total_counts, n_genes, pct_mt = streamed_qc(
        adata, mt_mask, row_mask=labelled, chunk_size=2048
    )

    atlas_obs_names = adata.obs_names[labelled]
    obs = adata.obs.loc[atlas_obs_names].copy()
    qc_cols = [c for c in ["sample_id", "condition", "author_cell_type"] if c in obs]
    qc = obs[qc_cols].copy()
    qc.insert(0, "nucleus_id", atlas_obs_names.astype(str))
    qc["n_genes_by_counts"] = n_genes
    qc["total_counts"] = total_counts
    qc["pct_counts_mt"] = pct_mt
    qc.to_csv(tables / "nucleus_qc_metrics.csv", index=False)

    if {"author_umap_1", "author_umap_2"}.issubset(obs.columns):
        save_categorical_umap(obs, figs / "author_umap_cell_types.png")

    genes = marker_panel(cfg, adata.var_names)
    target = cfg["project"]["target_gene"]
    load_genes = list(dict.fromkeys(genes + ([target] if target in adata.var_names else [])))
    print(f"Loading only {len(load_genes)} marker/target genes into RAM...", flush=True)
    loaded_genes, gx = small_gene_matrix(adata, load_genes, row_mask=labelled)
    pos = {g: i for i, g in enumerate(loaded_genes)}

    marker_genes = [g for g in genes if g in pos]
    if marker_genes:
        marker_x = gx[:, [pos[g] for g in marker_genes]]
        save_marker_dotplot(obs, marker_genes, marker_x, total_counts,
                            figs / "author_labels_marker_validation.png")

    if target in pos and {"author_umap_1", "author_umap_2"}.issubset(obs.columns):
        raw = gx[:, pos[target]]
        expr = np.log1p((raw / np.maximum(total_counts.astype(np.float32), 1.0)) * 1e4)
        save_target_umap(obs, expr, figs / f"{target}_author_umap.png", target)

    summary = pd.DataFrame({
        "metric": ["author_qc_nuclei", "genes", "median_total_counts", "median_genes", "median_pct_mt"],
        "value": [n_atlas, adata.n_vars, np.median(total_counts), np.median(n_genes), np.median(pct_mt)],
    })
    summary.to_csv(tables / "global_analysis_summary.csv", index=False)
    adata.file.close()
    print("Global analysis complete. Full count matrix was never loaded into RAM.", flush=True)


if __name__ == "__main__":
    main()
