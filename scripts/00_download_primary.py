#!/usr/bin/env python
from pathlib import Path

from pd_gpnmb.io import download, extract_tar
from pd_gpnmb.utils import load_config, ensure_dirs


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["paths"]["raw"]) / cfg["project"]["accession"]
    ensure_dirs(raw)
    base = cfg["primary"]["geo_base"]

    for key in ["clinical", "umap", "raw_tar"]:
        name = cfg["primary"]["files"][key]
        download(f"{base}/{name}", raw / name)

    extract_tar(raw / cfg["primary"]["files"]["raw_tar"], raw / "mtx")
    print("Primary dataset downloaded and extracted.")


if __name__ == "__main__":
    main()
