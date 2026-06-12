# GalactICSIsoWithGas — root Makefile
#
# Targets:
#   make install-dev   Python package + dev dependencies
#   make legacy-build  Compile legacy Fortran/C binaries
#   make test          Run pytest suite
#   make example-mw    Milky Way potential demo
#   make clean         Remove build artifacts

PYTHON ?= python3
VENV   ?= .venv
PIP    = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
PY     = $(VENV)/bin/python

.PHONY: all install-dev generate-artifacts legacy-build legacy-samplers legacy-clean test example-mw example-sample example-halo-first clean help

all: install-dev legacy-build

help:
	@echo "Targets:"
	@echo "  install-dev    Create .venv, install galacticsics + ntropy, generate test artifacts"
	@echo "  generate-artifacts  Run dbh+diskdf+sampling -> tests/generated/reference"
	@echo "  legacy-build   Build dbh -> legacy/bin/"
	@echo "  legacy-samplers Build gendisk, genhalo, genbulge, diskdf, getfreqs"
	@echo "  test           Run pytest"
	@echo "  example-mw     Run examples/mw_default.py"
	@echo "  example-solve  Run examples/solve_potential.py"
	@echo "  example-sample Run examples/sample_galaxy.py"
	@echo "  example-halo-first Run examples/halo_first_workflow.py"
	@echo "  clean          Remove .venv artifacts and legacy object files"

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"
	$(PIP) install -e "src/ntropy[dev]"

install-dev: $(VENV)/bin/activate generate-artifacts

generate-artifacts: legacy-build legacy-samplers
	$(PY) -m galacticsics.artifacts.cli generate

legacy-build:
	@if command -v make >/dev/null 2>&1; then \
		$(MAKE) -C legacy/fortran dbh && cp legacy/fortran/dbh legacy/bin/; \
	else \
		./scripts/build_legacy_nmake.sh; \
	fi

legacy-samplers:
	./scripts/build_samplers.sh

legacy-clean:
	$(MAKE) -C legacy/fortran clean

test: install-dev
	$(PYTEST) tests/ src/ntropy/tests/ -v --tb=short

example-mw: install-dev
	$(PY) examples/mw_default.py

example-solve: install-dev legacy-build
	$(PY) examples/solve_potential.py

example-sample: install-dev legacy-samplers
	$(PY) examples/sample_galaxy.py

example-halo-first: install-dev legacy-build legacy-samplers
	$(PY) examples/halo_first_workflow.py

clean: legacy-clean
	rm -rf $(VENV) build dist *.egg-info src/*.egg-info src/ntropy/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
