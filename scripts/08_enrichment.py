#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import gseapy as gp
import pandas as pd

from pd_gpnmb.utils import load_config, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("de_file")
    parser.add_argument("--label", default="microglia")
    parser.add_argument("--gene-set", default="GO_Biological_Process_2023")
    args = parser.parse_args()

    cfg = load_config()
    outdir = Path(cfg["paths"]["results"]) / "enrichment" / args.label
    ensure_dirs(outdir)

    de = pd.read_csv(args.de_file, index_col=0)
    de = de.dropna(subset=["log2FoldChange", "pvalue"]).copy()
    # Signed statistic keeps direction while prioritising well-supported genes.
    de["rank_score"] = de["log2FoldChange"] * (-de["pvalue"].clip(lower=1e-300).map(__import__("math").log10))
    ranking = de["rank_score"].sort_values(ascending=False)

    # Enrichr/GSEApy may need internet access to retrieve named libraries.
    pre = gp.prerank(
        rnk=ranking,
        gene_sets=args.gene_set,
        permutation_num=1000,
        seed=int(cfg["project"]["random_seed"]),
        outdir=str(outdir),
        verbose=True,
    )
    pre.res2d.to_csv(outdir / "gsea_results.csv", index=False)


if __name__ == "__main__":
    main()
