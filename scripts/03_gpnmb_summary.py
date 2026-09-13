#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from pd_gpnmb.backed import small_gene_matrix
from pd_gpnmb.plotting import donor_boxplot
from pd_gpnmb.utils import load_config, ensure_dirs


def main() -> None:
    cfg = load_config()
    accession = cfg["project"]["accession"]
    target = cfg["project"]["target_gene"]
    processed = Path(cfg["paths"]["processed"])
    results = Path(cfg["paths"]["results"])
    tables = results / "tables"
    figs = results / "figures"
    ensure_dirs(tables, figs)

    inp = processed / f"{accession}_raw_qc_annotated.h5ad"
    adata = ad.read_h5ad(inp, backed="r")
    required = ["sample_id", "condition", "author_cell_type"]
    missing = [c for c in required if c not in adata.obs]
    if missing:
        raise KeyError(f"Missing metadata columns: {missing}")
    if target not in adata.var_names:
        raise KeyError(f"{target} is not present in the matrix")

    qc_path = tables / "nucleus_qc_metrics.csv"
    if not qc_path.exists():
        raise FileNotFoundError("Run scripts/02_global_analysis.py first; nucleus_qc_metrics.csv is missing")
    qc = pd.read_csv(qc_path).set_index("nucleus_id")

    labelled = adata.obs["author_cell_type"].notna().to_numpy()
    atlas_names = adata.obs_names[labelled]
    obs = adata.obs.loc[atlas_names, required].copy()
    total = qc.reindex(atlas_names.astype(str))["total_counts"].to_numpy(float)
    if np.isnan(total).any():
        raise ValueError("QC table is not aligned to the annotated nuclei; rerun step 02")

    _, x = small_gene_matrix(adata, [target], row_mask=labelled)
    raw = x[:, 0]
    values = np.log1p((raw / np.maximum(total.astype(np.float32), 1.0)) * 1e4)

    cell = obs.copy()
    cell["expression_log1p"] = values
    cell["detected"] = raw > 0

    donor = (
        cell.groupby(["sample_id", "condition", "author_cell_type"], observed=True)
        .agg(
            mean_expression=("expression_log1p", "mean"),
            median_expression=("expression_log1p", "median"),
            detection_fraction=("detected", "mean"),
            n_nuclei=("detected", "size"),
        )
        .reset_index()
    )
    donor.to_csv(tables / f"{target}_donor_celltype_summary.csv", index=False)

    broad = (
        donor.groupby("author_cell_type", observed=True)
        .agg(
            donors=("sample_id", "nunique"),
            median_donor_expression=("mean_expression", "median"),
            median_detection_fraction=("detection_fraction", "median"),
            nuclei=("n_nuclei", "sum"),
        )
        .sort_values("median_donor_expression", ascending=False)
    )
    broad.to_csv(tables / f"{target}_celltype_overview.csv")

    micro_match = str(cfg["microglia"]["author_label_match"]).lower()
    micro = donor[donor["author_cell_type"].astype(str).str.lower().str.contains(micro_match, regex=False)].copy()
    if not micro.empty:
        donor_boxplot(micro, "mean_expression", "condition",
                      figs / f"{target}_microglia_donor_expression.png",
                      f"Mean log-normalised {target}")
        donor_boxplot(micro, "detection_fraction", "condition",
                      figs / f"{target}_microglia_detection_fraction.png",
                      f"Fraction of nuclei expressing {target}")

    adata.file.close()
    print(broad.head(10))


if __name__ == "__main__":
    main()
