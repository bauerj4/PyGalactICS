# GalactICSIsoWithGas — root Makefile
#
# Targets:
#   make install-dev   Python package + dev dependencies
#   make legacy-build  Compile legacy Fortran/C binaries
#   make test           Run full pytest suite
#   make test-essential PR gate (docs/ci_essential.md)
#   make loc            Lines of code (cloc; code / comment / blank)
#   make example-mw    Milky Way potential demo
#   make projections   Face-on / side-on component PNGs for a campaign
#   make projections-clean  Remove campaign projection PNGs
#   make clean         Remove build artifacts
#   make papers-zip    Zip MNRAS Part~1 manuscript + figures for Drive upload

PYTHON ?= python3
VENV   ?= .venv
PIP    = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
PY     = $(VENV)/bin/python
CLOC   ?= cloc

# Campaign projection PNGs (face-on / side-on per component)
CAMPAIGN_ROOT ?= runs/mw_morton_corpus_v2
PROJECTION_SNAPSHOTS ?=
PROJECTION_COMPONENTS ?=
PROJECTION_DPI ?= 120
PROJECTION_STEP_STRIDE ?= 1

# Extra cloc flags (e.g. LOC_ARGS='--csv' make loc)
LOC_ARGS ?=

.PHONY: all install-dev install-python-deps install-system-mpi generate-artifacts legacy-build legacy-samplers legacy-clean test test-essential loc cloc example-mw example-sample example-halo-first projections projections-clean papers-zip clean help

all: install-dev legacy-build

help:
	@echo "Targets:"
	@echo "  install-dev    Create .venv, install galacticsics + ntropy (mpi4py when OpenMPI available), generate test artifacts"
	@echo "  install-system-mpi  Install OpenMPI system packages for mpi4py (Debian/dnf/Homebrew)"
	@echo "  generate-artifacts  Run Python dbh+diskdf+sampling -> tests/generated/reference"
	@echo "  legacy-build   Build dbh -> legacy/bin/"
	@echo "  legacy-samplers Build gendisk, genhalo, genbulge, diskdf, getfreqs"
	@echo "  test           Run full pytest (excludes legacy_binary)"
	@echo "  test-essential Fast PR gate (-m essential; see docs/ci_essential.md)"
	@echo "  loc            Count LOC with cloc (code/comment/blank; skips .venv & gitignored)"
	@echo "  example-mw     Run examples/mw_default.py"
	@echo "  example-solve  Run examples/solve_potential.py"
	@echo "  example-sample Run examples/sample_galaxy.py"
	@echo "  example-halo-first Run examples/halo_first_workflow.py"
	@echo "  projections    Face-on/side-on PNGs per component under CAMPAIGN_ROOT"
	@echo "                 (default: ic + particle steps + final; PROJECTION_STEP_STRIDE=N thins steps)"
	@echo "  projections-clean  Remove */projections under CAMPAIGN_ROOT"
	@echo "  papers-zip     Bundle papers/mnras_noneq_ics (tex/pdf/figs/bib) -> dist/mnras_noneq_ics_bundle.zip"
	@echo "  clean          Remove .venv artifacts and legacy object files"

install-system-mpi:
	@bash scripts/ensure_openmpi.sh || true

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip

install-python-deps: $(VENV)/bin/python install-system-mpi
	$(PIP) install -e ".[dev]"
	$(PIP) install -e "src/ntropy[dev,mpi]"
	@if command -v mpirun >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libmpi\.so'; then \
		$(PIP) install --force-reinstall --no-cache-dir "mpi4py>=3.1" \
			|| { echo "ERROR: mpi4py rebuild failed. Install OpenMPI headers then retry:"; \
			     echo "  sudo apt-get install -y openmpi-bin libopenmpi-dev"; \
			     echo "  $(PIP) install --force-reinstall --no-cache-dir mpi4py"; exit 1; }; \
	else \
		echo "ERROR: OpenMPI (mpirun + libmpi) not found — MPI notebook cells will be skipped."; \
		echo "  sudo apt-get update && sudo apt-get install -y openmpi-bin libopenmpi-dev"; \
		echo "  $(PIP) install --force-reinstall --no-cache-dir mpi4py"; \
		exit 1; \
	fi
	@$(PY) -c "from mpi4py import MPI; print('mpi4py OK (COMM_WORLD size =', MPI.COMM_WORLD.Get_size(), ')')"
	@echo "mpirun: $$(mpirun --version 2>/dev/null | head -1)"

install-dev: install-python-deps generate-artifacts

generate-artifacts:
	$(PY) -m galacticsics.artifacts.cli generate

legacy-build:
	@if command -v make >/dev/null 2>&1; then \
		$(MAKE) -C legacy/fortran dbh && bash scripts/install_binary.sh legacy/fortran/dbh legacy/bin/dbh; \
	else \
		./scripts/build_legacy_nmake.sh; \
	fi

legacy-samplers:
	./scripts/build_samplers.sh

legacy-clean:
	$(MAKE) -C legacy/fortran clean

test: install-dev
	$(PYTEST) tests/ src/ntropy/tests/ -v --tb=short -m "not legacy_binary"

test-essential:
	$(PYTEST) tests/ src/ntropy/tests/ -v --tb=short -m "essential and not legacy_binary and not slow"

# Lines of code via cloc. Uses `git ls-files` so .venv, site-packages, build/,
# artifacts, and other gitignored install noise are excluded automatically.
loc cloc:
	@command -v $(CLOC) >/dev/null 2>&1 || { \
		echo "cloc not found. Install: sudo apt install cloc  (or brew install cloc)"; \
		exit 1; \
	}
	@echo "=== Summary (tracked sources; blank / comment / code) ==="
	@$(CLOC) --vcs=git $(LOC_ARGS) .
	@echo ""
	@echo "=== By area ==="
	@for d in src/galacticsics src/ntropy/ntropy tests src/ntropy/tests \
		docs examples scripts notebooks campaigns models; do \
		if [ -d "$$d" ]; then \
			echo ""; \
			echo "--- $$d ---"; \
			$(CLOC) --vcs=git --quiet $(LOC_ARGS) "$$d"; \
		fi; \
	done

example-mw: install-dev
	$(PY) examples/mw_default.py

example-solve: install-dev legacy-build
	$(PY) examples/solve_potential.py

example-sample: install-dev legacy-samplers
	$(PY) examples/sample_galaxy.py

example-halo-first: install-dev legacy-build legacy-samplers
	$(PY) examples/halo_first_workflow.py

# Face-on (x–y) and side-on (x–z) surface-density PNGs for disk/bulge/halo.
# Examples:
#   make projections
#   make projections CAMPAIGN_ROOT=runs/mw_morton_corpus
#   make projections PROJECTION_SNAPSHOTS='ic steps' PROJECTION_STEP_STRIDE=5
#   make projections PROJECTION_COMPONENTS='disk halo'
projections:
	@snap_args=""; \
	comp_args=""; \
	if [ -n "$(PROJECTION_SNAPSHOTS)" ]; then snap_args="--snapshots $(PROJECTION_SNAPSHOTS)"; fi; \
	if [ -n "$(PROJECTION_COMPONENTS)" ]; then comp_args="--components $(PROJECTION_COMPONENTS)"; fi; \
	$(PY) -m galacticsics.campaign.cli projections $(CAMPAIGN_ROOT) \
		$$snap_args $$comp_args --dpi $(PROJECTION_DPI) \
		--step-stride $(PROJECTION_STEP_STRIDE)

projections-clean:
	@if [ ! -d "$(CAMPAIGN_ROOT)" ]; then \
		echo "CAMPAIGN_ROOT=$(CAMPAIGN_ROOT) not found"; exit 1; \
	fi
	@n=$$(find "$(CAMPAIGN_ROOT)" -mindepth 2 -maxdepth 2 -type d -name projections 2>/dev/null | wc -l); \
	find "$(CAMPAIGN_ROOT)" -mindepth 2 -maxdepth 2 -type d -name projections -exec rm -rf {} + 2>/dev/null || true; \
	echo "Removed $$n projections/ dir(s) under $(CAMPAIGN_ROOT)"

# Bundle Part~1 manuscript for Google Drive (papers/ is gitignored).
# Includes tex/pdf/bib/figs/figures_eps + small results/; excludes archive/,
# run dumps, secrets. See BUNDLE_CONTENTS.md inside the zip.
papers-zip:
	@bash scripts/papers_zip.sh

clean: legacy-clean
	rm -rf $(VENV) build dist *.egg-info src/*.egg-info src/ntropy/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
