#!/usr/bin/env python
from pathlib import Path

import anndata as ad

from pd_gpnmb.utils import load_config, ensure_dirs


def main() -> None:
    cfg = load_config()
    accession = cfg["project"]["accession"]
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["results"]) / "tables"
    ensure_dirs(tables)

    adata = ad.read_h5ad(processed / f"{accession}_raw_qc_annotated.h5ad", backed="r")
    required = ["sample_id", "condition", "author_cell_type"]
    missing = [c for c in required if c not in adata.obs]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # obs is small metadata and is already memory-resident; X remains on disk.
    info = adata.obs[required].dropna().copy()
    info.columns = ["sample", "group", "cell_type"]
    info.to_csv(tables / "cell_composition_input.csv", index=False)

    counts = info.groupby(["sample", "group", "cell_type"], observed=True).size().rename("n_nuclei").reset_index()
    totals = counts.groupby("sample", observed=True)["n_nuclei"].transform("sum")
    counts["proportion"] = counts["n_nuclei"] / totals
    counts.to_csv(tables / "cell_composition_descriptive.csv", index=False)
    adata.file.close()
    print(counts.head())


if __name__ == "__main__":
    main()
