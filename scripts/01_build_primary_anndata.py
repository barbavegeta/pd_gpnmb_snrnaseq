#!/usr/bin/env python
from pathlib import Path

import anndata as ad

from pd_gpnmb.io import build_from_geo_triplets, read_author_umap, read_clinical, attach_metadata
from pd_gpnmb.utils import load_config, ensure_dirs


def main() -> None:
    cfg = load_config()
    accession = cfg["project"]["accession"]
    raw = Path(cfg["paths"]["raw"]) / accession
    processed = Path(cfg["paths"]["processed"])
    ensure_dirs(processed)

    # Checkpoint immediately after the expensive 29-sample sparse stack. If metadata
    # parsing fails later, the next run resumes here instead of rereading all MTX files.
    checkpoint = processed / f"{accession}_counts_checkpoint.h5ad"
    if checkpoint.exists():
        print(f"Loading checkpoint: {checkpoint}", flush=True)
        adata = ad.read_h5ad(checkpoint)
    else:
        adata = build_from_geo_triplets(raw / "mtx")
        print(f"Writing checkpoint: {checkpoint}", flush=True)
        adata.write_h5ad(checkpoint, compression="gzip")
        print(f"Checkpoint saved: {checkpoint}", flush=True)

    umap = read_author_umap(raw / cfg["primary"]["files"]["umap"])
    clinical = read_clinical(raw / cfg["primary"]["files"]["clinical"])
    adata = attach_metadata(adata, umap, clinical)

    # Keep raw counts in X in this file. Step 02 creates a dedicated counts layer
    # before normalisation, avoiding an unnecessary duplicate during the build step.
    out = processed / f"{accession}_raw_qc_annotated.h5ad"
    print(f"Writing: {out}", flush=True)
    adata.write_h5ad(out, compression="gzip")
    print(adata)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
