.PHONY: setup download build primary microglia de composition test

setup:
	conda env create -f environment.yml
	@echo "Then: conda activate pd-gpnmb-snrnaseq && pip install -e . --no-deps"

download:
	python scripts/00_download_primary.py

build:
	python scripts/01_build_primary_anndata.py

primary:
	python scripts/02_global_analysis.py
	python scripts/03_gpnmb_summary.py

microglia:
	python scripts/04_microglia_subcluster.py

de:
	python scripts/05_pseudobulk_de.py --group-col author_cell_type --group-match micro --label microglia

composition:
	python scripts/06_export_composition.py

# R/07_propeller.R is run separately once speckle is installed.
test:
	pytest -q
