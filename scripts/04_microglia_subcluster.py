#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import anndata as ad
import pandas as pd
import scanpy as sc

from pd_gpnmb.plotting import savefig
from pd_gpnmb.utils import load_config, ensure_dirs


def existing(genes: list[str], var_names) -> list[str]:
    return [g for g in genes if g in var_names]


def main() -> None:
    cfg = load_config()
    accession = cfg["project"]["accession"]
    processed = Path(cfg["paths"]["processed"])
    results = Path(cfg["paths"]["results"])
    figs = results / "figures"
    tables = results / "tables"
    ensure_dirs(figs, tables)

    inp = processed / f"{accession}_raw_qc_annotated.h5ad"
    backed = ad.read_h5ad(inp, backed="r")
    if "author_cell_type" not in backed.obs:
        raise KeyError("author_cell_type is missing")

    match = str(cfg["microglia"]["author_label_match"]).lower()
    mask = backed.obs["author_cell_type"].astype(str).str.lower().str.contains(match, regex=False).to_numpy()
    if mask.sum() == 0:
        raise ValueError(f"No author labels matched '{match}'")

    print(f"Loading only {mask.sum():,} microglial nuclei into RAM...", flush=True)
    micro = backed[mask].to_memory()
    backed.file.close()
    # This duplicate is now only for the microglial subset, not the 173k-cell atlas.
    micro.layers["counts"] = micro.X.copy()

    # Preserve integer counts above, then normalise/log-transform before
    # dispersion-based HVG selection.  ``seurat_v3`` uses a LOESS fit
    # (scikit-misc) which can become numerically singular on sparse
    # donor-stratified subsets such as these microglia.  The classic
    # Seurat dispersion method is stable here and still supports
    # batch-aware HVG selection via sample_id.
    sc.pp.normalize_total(micro, target_sum=1e4)
    sc.pp.log1p(micro)
    sc.pp.highly_variable_genes(
        micro,
        flavor="seurat",
        n_top_genes=min(int(cfg["microglia"]["n_hvg"]), micro.n_vars),
        batch_key="sample_id",
    )

    work = micro[:, micro.var["highly_variable"]].copy()
    sc.pp.scale(work, max_value=10, zero_center=False)
    sc.pp.pca(work, n_comps=min(int(cfg["microglia"]["n_pcs"]), work.n_vars - 1), zero_center=False)
    micro.obsm["X_pca"] = work.obsm["X_pca"]

    rep = "X_pca"
    if bool(cfg["microglia"].get("use_harmony", False)):
        import scanpy.external as sce
        sce.pp.harmony_integrate(micro, key="sample_id", basis="X_pca", adjusted_basis="X_pca_harmony")
        rep = "X_pca_harmony"

    sc.pp.neighbors(micro, use_rep=rep, n_neighbors=int(cfg["microglia"]["n_neighbors"]),
                    random_state=int(cfg["project"]["random_seed"]))
    sc.tl.umap(micro, random_state=int(cfg["project"]["random_seed"]))
    sc.tl.leiden(micro, resolution=float(cfg["microglia"]["leiden_resolution"]),
                 key_added="leiden_micro", random_state=int(cfg["project"]["random_seed"]))

    for name, genes in cfg["microglia_signatures"].items():
        genes = existing(genes, micro.var_names)
        if len(genes) >= 2:
            sc.tl.score_genes(micro, genes, score_name=f"score_{name}",
                              random_state=int(cfg["project"]["random_seed"]))

    target = cfg["project"]["target_gene"]
    plot_cols = ["leiden_micro", "condition"]
    if target in micro.var_names:
        plot_cols.append(target)
    plot_cols.extend([c for c in micro.obs.columns if c.startswith("score_")])
    sc.pl.umap(micro, color=plot_cols, show=False, frameon=False, ncols=2)
    savefig(figs / "microglia_umap_states.png")

    signature_genes: list[str] = []
    for genes in cfg["microglia_signatures"].values():
        signature_genes.extend(existing(genes, micro.var_names))
    signature_genes = list(dict.fromkeys(signature_genes))
    if signature_genes:
        sc.pl.dotplot(micro, signature_genes, groupby="leiden_micro", show=False, standard_scale="var")
        savefig(figs / "microglia_cluster_signature_dotplot.png")

    cluster_summary = (
        micro.obs.groupby(["leiden_micro", "condition"], observed=True)
        .agg(n_nuclei=("sample_id", "size"), n_donors=("sample_id", "nunique"))
        .reset_index()
    )
    score_cols = [c for c in micro.obs.columns if c.startswith("score_")]
    if score_cols:
        scores = micro.obs.groupby("leiden_micro", observed=True)[score_cols].mean().reset_index()
        cluster_summary = cluster_summary.merge(scores, on="leiden_micro", how="left")
    cluster_summary.to_csv(tables / "microglia_cluster_summary.csv", index=False)

    out = processed / f"{accession}_microglia_processed.h5ad"
    micro.write_h5ad(out, compression="gzip")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
