#!/usr/bin/env python
"""How many donors would the GPNMB microglial test need?

Simulates donor-level pseudobulk counts for the target gene from a negative binomial
calibrated to the observed DESeq2 result (baseMean and log2FC standard error under the
real `~ age + sex + condition` design), then re-tests PD vs control at increasing
cohort sizes. Two significance bars are reported:

- nominal: the pre-specified single-gene test (P < 0.05);
- transcriptome-wide FDR: the Benjamini-Hochberg cut-off the observed analysis actually
  used, P <= q * R / m, where R genes were rejected out of m tested. Larger cohorts
  reject more genes and loosen this bar, so the FDR curves are conservative.

Needs only numpy and pandas, so it runs without the processed AnnData.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from pd_gpnmb.power import backsolve_dispersion, design_matrix, simulate_power
from pd_gpnmb.utils import ensure_dirs, load_config

COLOURS = ["#5e9f96", "#1d7068", "#0a3c37"]  # sequential teal: small -> large effect
DASHES = ["2 4", "8 4", ""]


def bh_threshold(de: pd.DataFrame, fdr: float) -> tuple[float, int, int]:
    tested = de["padj"].notna()
    m = int(tested.sum())
    r = int((de.loc[tested, "padj"] <= fdr).sum())
    return fdr * max(r, 1) / m, r, m


def sci(value: float) -> str:
    """Format a small p-value as e.g. 4.9 × 10⁻⁴ for figure text."""
    mantissa, exponent = f"{value:.1e}".split("e")
    sup = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa} × 10{sup}"


def min_donors(curve: pd.DataFrame, column: str, target: float) -> str:
    hit = curve.loc[curve[column] >= target, "n_per_group"]
    return str(int(hit.min())) if len(hit) else f">{int(curve['n_per_group'].max())}"


def svg_figure(curves: pd.DataFrame, panels: list[tuple[str, str]], target: float, observed_n: float) -> str:
    """Two small-multiple power curves as a standalone SVG (no plotting dependencies)."""
    W, H = 1000, 470
    pw, ph = 360, 300
    left0, top = 80, 96
    gap = 150
    xmax = float(curves["n_per_group"].max())
    ink, ink2, ink3, grid = "#221c12", "#4b4232", "#6e6450", "#e6dfcd"
    effects = sorted(curves["true_log2fc"].unique())

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'font-family="IBM Plex Sans, Helvetica, Arial, sans-serif" role="img" '
        'aria-labelledby="t d">',
        '<title id="t">Power to detect higher GPNMB in PD microglia by donors per group</title>',
        '<desc id="d">Simulated power curves for true log2 fold changes of 0.5, 0.75 and 1.0, '
        'at the pre-specified P &lt; 0.05 and at the transcriptome-wide FDR cut-off.</desc>',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    # Legend, one row above both panels.
    lx = left0
    out.append(f'<text x="{lx}" y="40" font-size="15" fill="{ink2}">True effect in PD microglia:</text>')
    lx += 205
    for eff, col, dash in zip(effects, COLOURS, DASHES):
        out.append(f'<line x1="{lx}" y1="35" x2="{lx + 30}" y2="35" stroke="{col}" stroke-width="2.5" '
                   f'stroke-dasharray="{dash}"/>')
        out.append(f'<circle cx="{lx + 15}" cy="35" r="4" fill="{col}" stroke="#ffffff" stroke-width="2"/>')
        out.append(f'<text x="{lx + 38}" y="40" font-size="15" fill="{ink}">log2FC {eff:g} '
                   f'({2 ** eff:.2g}×)</text>')
        lx += 170

    for k, (column, title) in enumerate(panels):
        x0 = left0 + k * (pw + gap)

        def sx(v: float) -> float:
            return x0 + v / xmax * pw

        def sy(v: float) -> float:
            return top + (1 - v) * ph

        out.append(f'<text x="{x0}" y="{top - 22}" font-size="16" font-weight="600" fill="{ink}">{title}</text>')
        for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
            out.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + pw}" y2="{sy(v):.1f}" stroke="{grid}" stroke-width="1"/>')
            out.append(f'<text x="{x0 - 10}" y="{sy(v) + 5:.1f}" font-size="13" fill="{ink3}" text-anchor="end">{int(v * 100)}%</text>')
        for v in range(0, int(xmax) + 1, 20):
            out.append(f'<text x="{sx(v):.1f}" y="{top + ph + 22}" font-size="13" fill="{ink3}" text-anchor="middle">{v}</text>')
        out.append(f'<line x1="{x0}" y1="{sy(0):.1f}" x2="{x0 + pw}" y2="{sy(0):.1f}" stroke="{ink3}" stroke-width="1"/>')
        out.append(f'<text x="{x0 + pw / 2}" y="{top + ph + 48}" font-size="14" fill="{ink2}" text-anchor="middle">Donors per group</text>')
        # Target power and current cohort references.
        out.append(f'<line x1="{x0}" y1="{sy(target):.1f}" x2="{x0 + pw}" y2="{sy(target):.1f}" stroke="{ink2}" stroke-width="1" stroke-dasharray="4 3"/>')
        out.append(f'<text x="{x0 + pw - 4}" y="{sy(target) + 16:.1f}" font-size="12" fill="{ink2}" text-anchor="end">{int(target * 100)}% power</text>')
        out.append(f'<line x1="{sx(observed_n):.1f}" y1="{top}" x2="{sx(observed_n):.1f}" y2="{sy(0):.1f}" stroke="{ink3}" stroke-width="1" stroke-dasharray="2 3"/>')
        # Put the cohort label in whichever half the curves leave empty at this cohort size.
        nearest = curves.loc[(curves["n_per_group"] - observed_n).abs().idxmin(), "n_per_group"]
        high = curves.loc[curves["n_per_group"] == nearest, column].max() > 0.5
        label_y = sy(0) - 24 if high else top + 12
        out.append(f'<text x="{sx(observed_n) + 5:.1f}" y="{label_y:.1f}" font-size="12" fill="{ink3}">this study</text>')
        out.append(f'<text x="{sx(observed_n) + 5:.1f}" y="{label_y + 15:.1f}" font-size="12" fill="{ink3}">(14 vs 15)</text>')

        # Direct end labels, nudged apart so converging curves stay legible.
        ends = []
        for eff in effects:
            c = curves.loc[curves["true_log2fc"] == eff].sort_values("n_per_group").iloc[-1]
            ends.append([sy(c[column]) + 4, eff, sx(c["n_per_group"]) + 9])
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
        for y, eff, x in ends:
            out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="12" fill="{ink2}">{2 ** eff:.2g}×</text>')

        for eff, col, dash in zip(effects, COLOURS, DASHES):
            c = curves.loc[curves["true_log2fc"] == eff].sort_values("n_per_group")
            pts = " ".join(f"{sx(n):.1f},{sy(p):.1f}" for n, p in zip(c["n_per_group"], c[column]))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5" '
                       f'stroke-dasharray="{dash}" stroke-linejoin="round"/>')
            for n, p in zip(c["n_per_group"], c[column]):
                out.append(f'<circle cx="{sx(n):.1f}" cy="{sy(p):.1f}" r="4" fill="{col}" stroke="#ffffff" stroke-width="2"/>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    cfg = load_config()
    target_gene = cfg["project"]["target_gene"]
    pcfg = cfg["power"]
    results = Path(cfg["paths"]["results"])
    outdir = results / "power"
    ensure_dirs(outdir, results / "figures")

    de = pd.read_csv(results / "de" / "microglia_PD_vs_Control_pseudobulk.csv", index_col=0)
    donors = pd.read_csv(results / "de" / "microglia_eligible_donors.csv")
    obs = de.loc[target_gene]

    is_pd = (donors["condition"] == "PD").to_numpy()
    X = design_matrix(is_pd, donors["age"].to_numpy(), (donors["sex"] == "Male").to_numpy())
    dispersion = backsolve_dispersion(X, is_pd, obs["baseMean"], obs["log2FoldChange"], obs["lfcSE"])
    n_pd, n_ctrl = int(is_pd.sum()), int((~is_pd).sum())
    mu_ctrl = obs["baseMean"] * (n_pd + n_ctrl) / (n_ctrl + n_pd * 2.0 ** obs["log2FoldChange"])

    fdr_p, n_rejected, n_tested = bh_threshold(de, float(pcfg["fdr"]))
    thresholds = {"nominal": float(pcfg["nominal_alpha"]), "fdr": fdr_p}
    print(f"{target_gene}: baseMean {obs['baseMean']:.1f}, log2FC {obs['log2FoldChange']:.3f}, "
          f"SE {obs['lfcSE']:.3f}, implied NB dispersion {dispersion:.3f}")
    print(f"BH cut-off in the observed analysis: P <= {fdr_p:.2e} ({n_rejected} of {n_tested} genes at FDR {pcfg['fdr']})")

    rng = np.random.default_rng(int(cfg["project"]["random_seed"]))
    ctrl, pd_donors = donors.loc[~is_pd], donors.loc[is_pd]
    rows = []
    for lfc in [0.0] + [float(v) for v in pcfg["true_log2fc"]]:
        for n in pcfg["donors_per_group"]:
            rows.append(simulate_power(ctrl, pd_donors, int(n), lfc, mu_ctrl, dispersion,
                                       thresholds, int(pcfg["n_sims"]), rng))
    curves = pd.DataFrame(rows)
    curves.to_csv(outdir / f"{target_gene}_power_curves.csv", index=False)

    target = float(pcfg["target_power"])
    effects = curves.loc[curves["true_log2fc"] > 0]
    summary = []
    for lfc, c in effects.groupby("true_log2fc"):
        at15 = c.loc[c["n_per_group"] == 15].iloc[0]
        summary.append({
            "true_log2fc": lfc,
            "fold_change": round(2.0 ** lfc, 2),
            "power_at_15_nominal": at15["power_nominal"],
            "power_at_15_fdr": at15["power_fdr"],
            f"donors_per_group_for_{int(target * 100)}pct_nominal": min_donors(c, "power_nominal", target),
            f"donors_per_group_for_{int(target * 100)}pct_fdr": min_donors(c, "power_fdr", target),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(outdir / f"{target_gene}_power_summary.csv", index=False)

    null = curves.loc[curves["true_log2fc"] == 0, ["n_per_group", "power_nominal"]]
    meta = pd.Series({
        "target_gene": target_gene,
        "observed_log2fc": obs["log2FoldChange"],
        "observed_lfc_se": obs["lfcSE"],
        "observed_base_mean": obs["baseMean"],
        "implied_nb_dispersion": dispersion,
        "bh_p_cutoff": fdr_p,
        "bh_rejected": n_rejected,
        "bh_tested": n_tested,
        "type1_error_range": f"{null['power_nominal'].min():.3f}-{null['power_nominal'].max():.3f}",
        "n_sims": pcfg["n_sims"],
    })
    meta.to_csv(outdir / f"{target_gene}_power_parameters.csv", header=["value"])

    panels = [("power_nominal", "Pre-specified test (P &lt; 0.05)"),
              ("power_fdr", f"Transcriptome-wide FDR {pcfg['fdr']:g} (P ≤ {sci(fdr_p)})")]
    svg = svg_figure(effects, panels, target, observed_n=(n_pd + n_ctrl) / 2)
    (results / "figures" / f"{target_gene}_power_curves.svg").write_text(svg, encoding="utf-8")

    print(f"\nType I error at P < {pcfg['nominal_alpha']}: {meta['type1_error_range']} across cohort sizes")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
