.PHONY: all simulate analyze figures test clean

all: setup simulate analyze figures

# Set-up
setup: .setup_complete

.setup_complete: 
	@echo "Setting up environment"
	python3 -m pip install -r requirements.txt
	touch .setup_complete

# Run all scenarios and save raw results to results/raw/
simulate:
	PYTHONPATH=src python3 simulate.py

# Compute performance measures from results/raw/runs.csv
analyze:
	PYTHONPATH=src python3 -c "from metrics import run_all_metrics; run_all_metrics()"

# Create all visualizations
figures:
	PYTHONPATH=src python3 -c "from publication_figures import run_publication_figures; run_publication_figures()"
	PYTHONPATH=src python3 -c "from diagnostic_figures import run_diagnostic_figures; run_diagnostic_figures()"

# Run test suite
test:
	PYTHONPATH=src python3 -m pytest tests/

# Remove generated files
clean:
	rm -rf results/raw results/tables results/figures
	rm .*complete