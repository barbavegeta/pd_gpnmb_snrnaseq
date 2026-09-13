# GPNMB in Parkinson's disease: single-nucleus transcriptomic re-analysis

A focused re-analysis of human substantia nigra pars compacta (SNpc) single-nucleus RNA-seq data to determine where **GPNMB** is expressed, whether its microglial expression differs between Parkinson's disease (PD) and controls, and whether the direction of effect is supported by independent published transcriptomic analyses.

## Scientific question

**In which substantia nigra cell populations is GPNMB expressed, and how does its expression and associated microglial programme differ between Parkinson's disease and controls?**

This project complements genetic target-prioritisation evidence from Mendelian randomisation and colocalisation. Differential expression, MR and colocalisation are treated as distinct evidence layers and are not interpreted as interchangeable proof of causality.

## Primary dataset

**GSE243639 / Martirosyan et al.**

- 29 human post-mortem SNpc donors
- 15 sporadic PD
- 14 controls
- author-QC atlas used for the main analysis: **83,484 nuclei × 33,538 genes**

GEO provides filtered per-sample 10x MTX matrices, clinical metadata and author UMAP/cell annotations.

The authors had already applied nucleus-level QC, including doublet removal and mitochondrial-content filtering. This project therefore avoids imposing a second arbitrary round of filtering on the released author-QC dataset.

## Analysis principles

- The **donor**, not the nucleus, is the biological replicate.
- Raw integer counts are preserved for pseudobulk analysis.
- Cell-level expression plots are descriptive; formal PD-control DE uses donor-level pseudobulk counts.
- A minimum number of nuclei per donor/cell population is enforced before DE.
- Published broad cell labels are used for primary cell-type analyses and validated against marker expression.
- Microglia are re-clustered de novo to investigate GPNMB-associated cellular states.
- Harmony is **off by default**. Donor/library correction can be useful for exploratory embeddings but can also erase genuine donor biology; it is never used to create DE counts.
- Cell-type composition is tested with `speckle::propeller`.
- External validation is kept separate from the primary discovery analysis.
- GPNMB is a **pre-specified target gene**, not a post hoc hit selected from the transcriptome-wide DE results.

## Setup

The project was developed and run in the existing `bioinformatics` Conda environment.

```bash
conda activate bioinformatics
python -m pip install -e . --no-deps
```

Run commands from the repository root.

The editable install makes the `pd_gpnmb` helper package available to every Python pipeline script.

## Final pipeline

### 1. Download the primary GEO data

```bash
python scripts/00_download_primary.py
```

This downloads the clinical metadata, author UMAP/labels and sparse per-sample MTX archive.

The sparse MTX files are used rather than the large dense count table because they are much more memory-efficient.

### 2. Build the annotated AnnData object

```bash
python scripts/01_build_primary_anndata.py
```

Primary output:

```text
data/processed/GSE243639_raw_qc_annotated.h5ad
```

A disk-backed/count checkpoint is also retained to avoid repeating expensive reconstruction work.

### 3. Global analysis and GPNMB localisation

```bash
python scripts/02_global_analysis.py
python scripts/03_gpnmb_summary.py
```

The low-memory global analysis opens the full atlas in backed mode and loads only the genes required for marker validation and target-gene plots into RAM.

Main atlas size:

```text
83,484 nuclei × 33,538 genes
```

Median donor-level GPNMB expression was highest in microglia:

| Cell type | Donors | Median donor expression | Median detection fraction | Nuclei |
|---|---:|---:|---:|---:|
| Micro | 29 | 0.2890 | 0.1844 | 12,995 |
| OPC | 29 | 0.1732 | 0.1717 | 6,644 |
| VC | 29 | 0.0801 | 0.0620 | 1,560 |
| Astro | 29 | 0.0233 | 0.0169 | 20,710 |
| Neurons | 29 | 0.0213 | 0.0269 | 6,196 |
| Oligo | 29 | 0.0082 | 0.0075 | 35,032 |
| T cells | 26 | 0.0000 | 0.0000 | 347 |

### 4. De novo microglial state analysis

```bash
python scripts/04_microglia_subcluster.py
```

This loads only the **12,995 microglial nuclei** into RAM, performs microglial re-clustering and saves:

```text
data/processed/GSE243639_microglia_processed.h5ad
```

The analysis evaluates pre-specified programmes including:

- homeostatic microglia
- GPNMB/SPP1/TREM2-associated activation
- interferon response

A numeric Leiden cluster is not treated as a biological state by itself. Interpretation requires its broader marker/signature pattern and donor/condition representation.

### 5. Donor-aware microglial pseudobulk differential expression

```bash
python scripts/05_pseudobulk_de.py --group-col author_cell_type --group-match micro --label microglia
```

Eligible biological replicates:

```text
Control: 14 donors
PD:      15 donors
```

Primary design:

```text
~ age + sex + condition
```

Only covariates present without missing values are included. The model deliberately avoids overfitting a 29-donor cohort with every available clinical variable.

Primary GPNMB result:

| Gene | baseMean | log2FC PD vs Control | SE | P-value | FDR |
|---|---:|---:|---:|---:|---:|
| GPNMB | 169.57 | **+1.025** | 0.390 | **0.00863** | 0.218 |

This corresponds to approximately:

```text
2^1.025 ≈ 2.03-fold higher GPNMB expression in PD microglial pseudobulk
```

The direction and nominal P-value support higher GPNMB expression in PD microglia, but the association does **not** remain significant after transcriptome-wide FDR correction.

Full output:

```text
results/de/microglia_PD_vs_Control_pseudobulk.csv
```

### 6. Cell-type composition

Export donor-level cell proportions:

```bash
python scripts/06_export_composition.py
```

Run Propeller:

```bash
Rscript R/07_propeller.R results/tables/cell_composition_input.csv results/tables/propeller_results.csv
```

Selected results:

| Cell type | Mean Control proportion | Mean PD proportion | P-value | FDR |
|---|---:|---:|---:|---:|
| Neurons | 0.0879 | 0.0402 | 0.00885 | 0.06195 |
| Astro | 0.2277 | 0.2932 | 0.0579 | 0.2028 |
| Micro | 0.1579 | 0.1641 | 0.5870 | 0.6848 |

The microglial proportion itself was not detectably different between PD and controls. Therefore, the higher GPNMB pseudobulk signal is not simply explained by PD donors having a larger broad microglial fraction.

The neuronal proportion difference was nominally strong but did not cross the conventional FDR < 0.05 threshold.

### 7. Pathway enrichment

```bash
python scripts/08_enrichment.py results/de/microglia_PD_vs_Control_pseudobulk.csv --label microglia
```

Outputs are written under:

```text
results/enrichment/microglia/
```

This analysis is secondary/exploratory and is interpreted separately from the pre-specified GPNMB test.

### 8. External transcriptomic validation

```bash
python scripts/09_external_validation_gse184950.py
```

A full in-house reconstruction of GSE184950 was explored but not used in the final workflow because the publicly available Synapse Seurat object lacks the final published cell-type/cluster labels required for a defensible microglial pseudobulk reconstruction.

The final validation step therefore uses published external PD microglial analyses and extracts the GPNMB result directly from their supplementary result tables.

Outputs:

```text
results/replication_external/wang_microglia_mast.csv
results/replication_external/combined_microglia_pseudobulk.csv
results/replication_external/GPNMB_external_validation_GSE184950.csv
```

Cross-dataset GPNMB summary:

| Dataset / analysis | Method | log2FC | P-value | FDR / adjusted P | Interpretation |
|---|---|---:|---:|---:|---|
| GSE243639 | donor-level pseudobulk DESeq2 | **+1.025** | **0.00863** | 0.218 | positive, nominally significant |
| GSE184950 / Wang | donor-aware microglial MAST | +0.113 | 0.131 | 1.000 | same direction, not significant |
| Wang + Kamath + Smajic | multi-dataset microglial pseudobulk | **+0.933** | **0.00186** | **0.050054** | concordant effect; borderline 5% FDR |

The combined pseudobulk effect corresponds to approximately:

```text
2^0.933 ≈ 1.91-fold higher expression
```

The GSE184950 result alone does **not** constitute a statistically significant independent replication of GPNMB. The appropriate interpretation is directional concordance in GSE184950 and a closely matched effect size in the independent multi-dataset pseudobulk analysis.

## Main interpretation

The primary GSE243639 analysis places GPNMB most strongly in the microglial compartment and identifies approximately two-fold higher donor-level microglial GPNMB expression in PD.

The single-dataset GSE184950 result is directionally concordant but not statistically significant. A separate multi-cohort microglial pseudobulk analysis reports a similar positive effect size (`logFC = 0.933`) with `P = 0.00186` and an FDR of `0.050054`, just above the conventional 0.05 threshold.

Together, these findings support **cross-study directional consistency with variable statistical strength**.

They provide transcriptomic cell-type and disease-state context for the genetic prioritisation of GPNMB from MR and colocalisation, but they do not establish that increased GPNMB expression causes Parkinson's disease or that lowering GPNMB would be therapeutic.

## Pre-specified primary analysis

Before inspecting disease-effect results, the primary analysis was defined as:

1. Use author broad labels for primary cell-type comparisons.
2. Define microglia as author cell types containing `micro`.
3. Require at least 30 nuclei per donor for microglial pseudobulk.
4. Require at least 5 eligible donors in each condition.
5. Primary DE design: `~ age + sex + condition` when those covariates are complete.
6. Primary contrast: PD versus Control.
7. Primary target gene: GPNMB.
8. Treat genome-wide DE and pathway analyses as secondary/exploratory.
9. Keep external validation separate from discovery.

These choices are stored in `config.yaml`, so threshold/model changes remain auditable in version control.

## Key outputs

### Figures

```text
results/figures/author_umap_cell_types.png
results/figures/author_labels_marker_validation.png
results/figures/GPNMB_author_umap.png
results/figures/GPNMB_microglia_donor_expression.png
results/figures/GPNMB_microglia_detection_fraction.png
results/figures/microglia_umap_states.png
results/figures/microglia_cluster_signature_dotplot.png
```

### Tables

```text
results/tables/global_analysis_summary.csv
results/tables/GPNMB_celltype_overview.csv
results/tables/GPNMB_donor_celltype_summary.csv
results/tables/microglia_cluster_summary.csv
results/tables/cell_composition_input.csv
results/tables/cell_composition_descriptive.csv
results/tables/propeller_results.csv
results/tables/gpnmb_cross_dataset_summary.csv
```

### Differential expression and enrichment

```text
results/de/microglia_PD_vs_Control_pseudobulk.csv
results/de/microglia_eligible_donors.csv
results/enrichment/microglia/
```

### External validation

```text
results/replication_external/GPNMB_external_validation_GSE184950.csv
```

## Final execution order

From the repository root:

```bash
python scripts/00_download_primary.py
python scripts/01_build_primary_anndata.py
python scripts/02_global_analysis.py
python scripts/03_gpnmb_summary.py
python scripts/04_microglia_subcluster.py
python scripts/05_pseudobulk_de.py --group-col author_cell_type --group-match micro --label microglia
python scripts/06_export_composition.py
Rscript R/07_propeller.R results/tables/cell_composition_input.csv results/tables/propeller_results.csv
python scripts/08_enrichment.py results/de/microglia_PD_vs_Control_pseudobulk.csv --label microglia
python scripts/09_external_validation_gse184950.py
```

## Tests

```bash
pytest -q
```

## Reproducibility

The repository includes:

```text
environment.yml
requirements_frozen.txt
pyproject.toml
config.yaml
```

Large raw and processed data files should not be committed to Git. Credentials or authentication tokens must never be stored in the repository.

## What not to claim

This project can identify cell-type/state-specific transcriptomic associations with PD and assess whether a pre-specified GPNMB signal is directionally consistent across datasets.

It cannot by itself establish:

- that GPNMB causes Parkinson's disease;
- that increased GPNMB expression is harmful or protective;
- that a specific microglial state mediates a genetic association;
- that GSE184950 alone significantly replicated the GPNMB association;
- that the published external multi-dataset analysis was performed de novo within this repository.

Those questions require separate causal evidence and careful integration with the MR/colocalisation results.
