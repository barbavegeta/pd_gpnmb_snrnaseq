from __future__ import annotations

import csv
import gzip
import shutil
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.io import mmread
from tqdm import tqdm

from .utils import (
    canonical_cell_id,
    canonical_condition,
    canonical_sample_id,
    canonical_sex,
    first_existing,
    sample_from_cell_id,
    snake_case,
)


def download(url: str, destination: str | Path, chunk_size: int = 1024 * 1024) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {destination} already exists")
        return destination

    tmp = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, desc=destination.name
        ) as bar:
            for block in response.iter_content(chunk_size=chunk_size):
                if block:
                    handle.write(block)
                    bar.update(len(block))
    tmp.replace(destination)
    return destination


def extract_tar(tar_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".extracted"
    if marker.exists():
        print(f"[skip] {tar_path} already extracted")
        return
    with tarfile.open(tar_path, "r:*") as archive:
        archive.extractall(output_dir, filter="data")
    marker.touch()


def _read_tsv_gz(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, compression="gzip")


def read_prefixed_10x_triplet(matrix_path: Path, features_path: Path, barcodes_path: Path) -> ad.AnnData:
    """Read a GEO 10x triplet whose filenames contain a sample-specific prefix."""
    stem = matrix_path.name.replace("_matrix.mtx.gz", "")
    # e.g. GSM7792123_s_0096 -> s_0096
    sample_bits = stem.split("_")
    sample_id = canonical_sample_id("_".join(sample_bits[-2:]))

    with gzip.open(matrix_path, "rb") as handle:
        x = mmread(handle).tocsr().T  # 10x: genes x cells -> AnnData: cells x genes

    features = _read_tsv_gz(features_path)
    barcodes = _read_tsv_gz(barcodes_path)[0].astype(str).tolist()
    gene_ids = features.iloc[:, 0].astype(str).tolist()
    gene_names = features.iloc[:, 1].astype(str).tolist() if features.shape[1] > 1 else gene_ids

    obs_names = [canonical_cell_id(f"{sample_id}_{bc}") for bc in barcodes]
    obs = pd.DataFrame(index=pd.Index(obs_names, name="cell_id"))
    obs["sample_id"] = sample_id
    var = pd.DataFrame({"gene_id": gene_ids}, index=pd.Index(gene_names, name="gene"))

    obj = ad.AnnData(X=x, obs=obs, var=var)
    obj.var_names_make_unique()
    return obj


def build_from_geo_triplets(extracted_dir: str | Path) -> ad.AnnData:
    """Build one AnnData object from the per-donor GEO 10x matrices.

    The original implementation kept 29 full AnnData objects in memory and then
    called ``anndata.concat``.  For this dataset that can create a large temporary
    allocation at concatenation time.  All samples were generated with the same
    feature table, so we validate that explicitly and sparse-stack the matrices
    instead.
    """
    extracted_dir = Path(extracted_dir)
    matrices = sorted(extracted_dir.rglob("*_matrix.mtx.gz"))
    if not matrices:
        raise FileNotFoundError(f"No *_matrix.mtx.gz files found under {extracted_dir}")

    x_blocks: list[sparse.csr_matrix] = []
    obs_blocks: list[pd.DataFrame] = []
    template_var: pd.DataFrame | None = None
    template_var_names: pd.Index | None = None

    total = len(matrices)
    for i, matrix_path in enumerate(matrices, start=1):
        prefix = matrix_path.name.replace("_matrix.mtx.gz", "")
        features_path = matrix_path.with_name(prefix + "_features.tsv.gz")
        barcodes_path = matrix_path.with_name(prefix + "_barcodes.tsv.gz")
        if not features_path.exists() or not barcodes_path.exists():
            raise FileNotFoundError(f"Missing features/barcodes for {matrix_path.name}")

        print(f"Reading {prefix} ({i}/{total})", flush=True)
        obj = read_prefixed_10x_triplet(matrix_path, features_path, barcodes_path)

        if template_var is None:
            template_var = obj.var.copy()
            template_var_names = obj.var_names.copy()
        elif not obj.var_names.equals(template_var_names):
            raise ValueError(
                f"Feature names/order differ in {matrix_path.name}; "
                "cannot safely sparse-stack samples."
            )

        x_blocks.append(sparse.csr_matrix(obj.X))
        obs_blocks.append(obj.obs.copy())

    assert template_var is not None
    print(f"Sparse-stacking {total} samples...", flush=True)
    x = sparse.vstack(x_blocks, format="csr")
    obs = pd.concat(obs_blocks, axis=0)

    combined = ad.AnnData(X=x, obs=obs, var=template_var)
    combined.obs_names_make_unique()
    print(f"Combined matrix: {combined.n_obs:,} nuclei x {combined.n_vars:,} genes", flush=True)
    return combined


def read_author_umap(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    cell_col = first_existing(df.columns, ["CELL_ID", "cell", "cell_id", "barcode"])
    if cell_col is None:
        raise ValueError(f"Cannot identify cell ID column in {path}. Columns: {list(df.columns)}")
    df["cell_key"] = df[cell_col].map(canonical_cell_id)
    df = df.drop_duplicates("cell_key").set_index("cell_key")
    rename = {}
    for old, new in [("IDENT", "author_cell_type"), ("CLUSTER", "author_cluster"), ("UMAP_1", "author_umap_1"), ("UMAP_2", "author_umap_2")]:
        col = first_existing(df.columns, [old])
        if col:
            rename[col] = new
    return df.rename(columns=rename)


def _read_ragged_csv(path: str | Path) -> list[list[str]]:
    """Read the GEO clinical metadata despite its prose preamble and delimiter switch.

    GSE243639_Clinical_data.csv.gz starts with free-text explanatory lines containing
    commas, then the actual donor table begins with a semicolon-delimited header
    (``N;Brain Bank ID;Sample ID;Clinical diagnosis;...``).  Reading the whole file
    as comma-separated data therefore destroys the real table.  Detect that header
    from the raw lines and parse the table section with ``;``.  Fall back to ordinary
    comma-separated ragged parsing for any future file that does not have this form.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        lines = handle.readlines()

    header_i = None
    for i, line in enumerate(lines):
        lowered = line.lower()
        if (
            "sample id" in lowered
            and "clinical diagnosis" in lowered
            and line.count(";") >= 5
        ):
            header_i = i
            break

    if header_i is not None:
        return [
            [cell.strip() for cell in row]
            for row in csv.reader(lines[header_i:], delimiter=";")
            if any(str(cell).strip() for cell in row)
        ]

    return [
        [cell.strip() for cell in row]
        for row in csv.reader(lines)
        if any(str(cell).strip() for cell in row)
    ]


def _clinical_table_from_rows(rows: list[list[str]]) -> pd.DataFrame:
    """Recover either a normal donor-by-row table or a transposed donor-by-column table."""
    rows = [row for row in rows if any(str(x).strip() for x in row)]
    if not rows:
        raise ValueError("Clinical metadata file is empty")

    sample_aliases = {"sample", "sample_id", "sampleid", "donor", "donor_id", "id", "brain_id"}
    metadata_aliases = {
        "clinical_diagnosis", "diagnosis", "condition", "disease", "group",
        "age", "age_years", "age_at_death", "sex", "gender",
        "pmi_hours", "pmi", "post_mortem_interval", "postmortem_interval",
        "rin_measure", "rin", "rna_integrity_number",
        "braak_stage_for_neurofibrillary_tangles", "braak_stage", "braak",
        "cerad_score_for_neuritic_plaques", "cerad_score", "cerad",
    }

    # 1) Conventional table: locate the real header even if the file has title/preamble rows.
    best = None
    for i, row in enumerate(rows):
        norm = [snake_case(x) for x in row]
        has_sample = any(x in sample_aliases for x in norm)
        n_meta = sum(x in metadata_aliases for x in norm)
        score = (10 if has_sample else 0) + n_meta
        if best is None or score > best[0]:
            best = (score, i, row)

    if best is not None and best[0] >= 11:  # sample column + at least one recognised metadata field
        _, header_i, header = best
        width = len(header)
        records = []
        for row in rows[header_i + 1:]:
            if len(row) < width:
                row = row + [""] * (width - len(row))
            elif len(row) > width:
                # Preserve the declared schema; any surplus malformed fields are appended
                # to the final column rather than shifting all earlier fields.
                row = row[: width - 1] + [",".join(row[width - 1:])]
            records.append(row)
        df = pd.DataFrame(records, columns=header)
        sample_col = first_existing(df.columns, sample_aliases)
        if sample_col is not None:
            valid = df[sample_col].astype(str).map(canonical_sample_id).str.match(r"^s_\d+$", na=False)
            df = df.loc[valid].copy()
            if not df.empty:
                return df

    # 2) Transposed table: find a row containing multiple sample IDs, then use those
    # sample columns while treating the first non-sample column as the field name.
    sample_row_i = None
    sample_positions: list[int] = []
    for i, row in enumerate(rows):
        positions = [j for j, value in enumerate(row) if canonical_sample_id(value).startswith("s_")]
        if len(positions) >= 5:
            sample_row_i = i
            sample_positions = positions
            break

    if sample_row_i is not None:
        sample_row = rows[sample_row_i]
        sample_ids = [canonical_sample_id(sample_row[j]) for j in sample_positions]
        records = {sid: {"sample_id": sid} for sid in sample_ids}
        first_sample_pos = min(sample_positions)

        for i, row in enumerate(rows):
            if i == sample_row_i:
                continue
            label_cells = [x for x in row[:first_sample_pos] if str(x).strip()]
            if not label_cells:
                continue
            field = snake_case(label_cells[-1])
            if not field:
                continue
            for sid, pos in zip(sample_ids, sample_positions):
                records[sid][field] = row[pos] if pos < len(row) else ""

        df = pd.DataFrame(records.values())
        if len(df) >= 5:
            return df

    # 3) Last-resort ragged frame, mainly to produce a useful diagnostic rather than
    # another cryptic pandas tokenisation error.
    max_width = max(len(r) for r in rows)
    preview = [r + [""] * (max_width - len(r)) for r in rows[:8]]
    raise ValueError(
        "Could not identify donor-level clinical metadata in the GEO clinical CSV. "
        f"First parsed rows: {preview}"
    )


def read_clinical(path: str | Path) -> pd.DataFrame:
    rows = _read_ragged_csv(path)
    df = _clinical_table_from_rows(rows)
    df.columns = [snake_case(c) for c in df.columns]

    sample_col = first_existing(
        df.columns,
        ["sample", "sample_id", "sampleid", "donor", "donor_id", "id", "brain_id"],
    )
    if sample_col is None:
        sample_col = df.columns[0]

    aliases = {
        "condition": ["clinical_diagnosis", "diagnosis", "condition", "disease", "group"],
        "age": ["age", "age_years", "age_at_death"],
        "sex": ["sex", "gender"],
        "pmi_hours": ["pmi_hours", "pmi", "post_mortem_interval", "postmortem_interval"],
        "rin": ["rin_measure", "rin", "rna_integrity_number"],
        "braak_stage": ["braak_stage_for_neurofibrillary_tangles", "braak_stage", "braak"],
        "cerad_score": ["cerad_score_for_neuritic_plaques", "cerad_score", "cerad"],
    }

    out = pd.DataFrame(index=df.index)
    out["sample_id"] = df[sample_col].map(canonical_sample_id)
    for standard, choices in aliases.items():
        col = first_existing(df.columns, choices)
        if col is not None:
            out[standard] = df[col]

    if "condition" in out:
        out["condition"] = out["condition"].map(canonical_condition)
    if "sex" in out:
        out["sex"] = out["sex"].map(canonical_sex)
    for numeric in ["age", "pmi_hours", "rin", "braak_stage", "cerad_score"]:
        if numeric in out:
            out[numeric] = pd.to_numeric(out[numeric], errors="coerce")

    out = out[out["sample_id"].astype(str).str.match(r"^s_\d+$", na=False)]
    out = out.drop_duplicates("sample_id").set_index("sample_id")

    if out.empty:
        raise ValueError("Clinical metadata parser produced zero valid sample IDs")
    if "condition" not in out.columns:
        raise ValueError(f"Clinical metadata contains no diagnosis/condition column. Columns: {list(df.columns)}")

    print(f"Clinical metadata: {len(out)} donors; conditions={out['condition'].value_counts(dropna=False).to_dict()}")
    return out


def attach_metadata(adata: ad.AnnData, umap: pd.DataFrame, clinical: pd.DataFrame) -> ad.AnnData:
    # Modify in place: copying the full sparse count matrix here is unnecessary
    # and can substantially increase peak RAM usage.
    adata.obs["cell_key"] = [canonical_cell_id(x) for x in adata.obs_names]
    adata.obs["sample_id"] = [sample_from_cell_id(x) for x in adata.obs_names]

    aligned_umap = umap.reindex(adata.obs["cell_key"])
    for col in ["author_cell_type", "author_cluster", "author_umap_1", "author_umap_2"]:
        if col in aligned_umap:
            adata.obs[col] = aligned_umap[col].to_numpy()

    for col in clinical.columns:
        adata.obs[col] = adata.obs["sample_id"].map(clinical[col])

    if {"author_umap_1", "author_umap_2"}.issubset(adata.obs.columns):
        coords = adata.obs[["author_umap_1", "author_umap_2"]].to_numpy(dtype=float)
        if np.isfinite(coords).all():
            adata.obsm["X_umap_author"] = coords

    matched = adata.obs.get("author_cell_type", pd.Series(index=adata.obs_names, dtype=object)).notna().mean()
    print(f"Matched author broad cell labels for {matched:.1%} of nuclei")
    return adata
