#!/usr/bin/env python
"""Download the independent GSE184950 processed-data archive.

This deliberately stops at data preparation. Cell-label harmonisation across studies must be
reviewed before formal replication; the primary analysis must not silently impose labels from
GSE243639 onto a different atlas.
"""
from pathlib import Path

from pd_gpnmb.io import download, extract_tar
from pd_gpnmb.utils import ensure_dirs

ACCESSION = "GSE184950"
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE184nnn/GSE184950/suppl"


def main() -> None:
    raw = Path("data/raw") / ACCESSION
    ensure_dirs(raw)
    raw_tar = download(f"{BASE}/{ACCESSION}_RAW.tar", raw / f"{ACCESSION}_RAW.tar")
    download(f"{BASE}/{ACCESSION}_add2.xlsx", raw / f"{ACCESSION}_add2.xlsx")
    extract_tar(raw_tar, raw / "processed_samples")
    print(
        "Replication archive prepared. Inspect GSE184950 annotations and harmonise broad cell "
        "types before running the same donor-aware GPNMB tests."
    )


if __name__ == "__main__":
    main()
