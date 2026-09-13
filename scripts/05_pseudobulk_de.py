#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from pd_gpnmb.pseudobulk import aggregate_sparse_counts, filter_pseudobulks_by_cell_number
from pd_gpnmb.utils import load_config, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Donor-aware pseudobulk differential expression.")
    parser.add_argument("--adata", default=None)
    parser.add_argument("--group-col", default="author_cell_type")
    parser.add_argument("--group-match", default="micro")
    parser.add_argument("--label", default="microglia")
    args = parser.parse_args()

    cfg = load_config()
    accession = cfg["project"]["accession"]
    processed = Path(cfg["paths"]["processed"])
    outdir = Path(cfg["paths"]["results"]) / "de"
    ensure_dirs(outdir)

    input_path = Path(args.adata) if args.adata else processed / f"{accession}_raw_qc_annotated.h5ad"
    backed = ad.read_h5ad(input_path, backed="r")
    if args.group_col not in backed.obs:
        raise KeyError(f"{args.group_col} is not in adata.obs")

    mask = backed.obs[args.group_col].astype(str).str.lower().str.contains(args.group_match.lower(), regex=False).to_numpy()
    if mask.sum() == 0:
        raise ValueError(f"No cells matched {args.group_col} contains '{args.group_match}'")
    print(f"Loading only {mask.sum():,} matched nuclei for pseudobulk...", flush=True)
    sub = backed[mask].to_memory()
    backed.file.close()

    min_nuclei = int(cfg["analysis"]["min_nuclei_per_pseudobulk"])
    eligible = filter_pseudobulks_by_cell_number(sub.obs, ["sample_id"], min_nuclei)
    eligible_samples = set(eligible["sample_id"].astype(str))
    sub = sub[sub.obs["sample_id"].astype(str).isin(eligible_samples)].copy()

    donor_counts = sub.obs.groupby("condition", observed=True)["sample_id"].nunique()
    min_donors = int(cfg["analysis"]["min_donors_per_condition"])
    print("Eligible donors by condition:")
    print(donor_counts)
    if any(donor_counts.get(group, 0) < min_donors for group in ["PD", "Control"]):
        raise ValueError(f"Need at least {min_donors} eligible donors in each condition")

    # X is raw counts in GSE243639_raw_qc_annotated.h5ad.
    counts, meta = aggregate_sparse_counts(sub.X, sub.obs, sub.var_names, group_columns=["sample_id"])
    clinical_cols = [c for c in ["condition", "age", "sex", "pmi_hours", "rin", "braak_stage"] if c in sub.obs]
    donor_meta = sub.obs[["sample_id"] + clinical_cols].drop_duplicates("sample_id").set_index("sample_id")
    meta = meta.join(donor_meta, on="sample_id", how="left")
    meta.index = counts.index

    counts = counts.sparse.to_dense().round().astype(int)
    min_count = int(cfg["analysis"]["min_gene_count"])
    min_samples = int(cfg["analysis"]["min_samples_gene_count"])
    keep = (counts >= min_count).sum(axis=0) >= min_samples
    counts = counts.loc[:, keep]

    covariates = []
    for cov in cfg["analysis"].get("de_covariates", []):
        if cov in meta.columns and meta[cov].notna().all() and meta[cov].nunique(dropna=True) > 1:
            covariates.append(cov)
    design = "~ " + " + ".join(covariates + ["condition"])
    print(f"DE design: {design}")

    for cov in covariates:
        if cov in {"age", "pmi_hours", "rin", "braak_stage"}:
            meta[cov] = pd.to_numeric(meta[cov], errors="raise")
        else:
            meta[cov] = meta[cov].astype("category")
    meta["condition"] = pd.Categorical(meta["condition"], categories=["Control", "PD"])

    dds = DeseqDataSet(counts=counts, metadata=meta, design=design, refit_cooks=True, n_cpus=1, quiet=False)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["condition", "PD", "Control"], alpha=0.05, n_cpus=1)
    stats.summary()
    res = stats.results_df.copy().sort_values("padj", na_position="last")
    res.index.name = "gene"
    res.to_csv(outdir / f"{args.label}_PD_vs_Control_pseudobulk.csv")

    target = cfg["project"]["target_gene"]
    if target in res.index:
        print("\nTarget gene result:")
        print(res.loc[[target]])

    audit = meta.copy()
    nuclei_per_donor = sub.obs.groupby("sample_id", observed=True).size().to_dict()
    audit["n_nuclei"] = audit["sample_id"].map(nuclei_per_donor)
    audit.to_csv(outdir / f"{args.label}_eligible_donors.csv", index=True)


if __name__ == "__main__":
    main()
