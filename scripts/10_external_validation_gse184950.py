#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import io
import requests
import pandas as pd

OUTDIR = Path("results/replication_external")
OUTDIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "wang_microglia_mast": (
        "https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1186%2Fs12863-026-01462-2/MediaObjects/"
        "12863_2026_1462_MOESM3_ESM.csv"
    ),
    "combined_microglia_pseudobulk": (
        "https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1186%2Fs12863-026-01462-2/MediaObjects/"
        "12863_2026_1462_MOESM5_ESM.csv"
    ),
}

def read_csv_flex(content: bytes) -> pd.DataFrame:
    # Try the ordinary parser first, then sniff separators.
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python")

def find_gpnmb(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        c for c in df.columns
        if any(k in str(c).lower() for k in ("gene", "symbol", "feature"))
    ]
    if not candidates:
        candidates = list(df.columns)

    mask = pd.Series(False, index=df.index)
    for c in candidates:
        vals = df[c].astype(str).str.strip().str.upper()
        mask |= vals.eq("GPNMB")
    return df.loc[mask].copy()

summary_rows = []

for label, url in FILES.items():
    print(f"\nDownloading {label} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    raw_path = OUTDIR / f"{label}.csv"
    raw_path.write_bytes(r.content)

    df = read_csv_flex(r.content)
    print(f"Rows x columns: {df.shape[0]:,} x {df.shape[1]}")
    print("Columns:", ", ".join(map(str, df.columns)))

    hit = find_gpnmb(df)
    if hit.empty:
        print("GPNMB not found automatically.")
        print("First rows:")
        print(df.head().to_string(index=False))
    else:
        print("\nGPNMB result:")
        print(hit.to_string(index=False))
        hit.insert(0, "source_analysis", label)
        summary_rows.append(hit)

if summary_rows:
    summary = pd.concat(summary_rows, ignore_index=True, sort=False)
    out = OUTDIR / "GPNMB_external_validation_GSE184950.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved GPNMB validation summary: {out}")
else:
    print("\nNo GPNMB rows found. Inspect the downloaded supplementary CSV files.")
