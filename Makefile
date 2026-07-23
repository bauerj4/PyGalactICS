# GalactICSIsoWithGas — root Makefile
#
# Targets:
#   make install-dev   Python package + dev dependencies
#   make legacy-build  Compile legacy Fortran/C binaries
#   make test           Run full pytest suite
#   make test-essential PR gate (docs/ci_essential.md)
#   make loc            Lines of code (cloc; code / comment / blank)
#   make example-mw    Milky Way potential demo
#   make clean         Remove build artifacts

PYTHON ?= python3
VENV   ?= .venv
PIP    = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
PY     = $(VENV)/bin/python
CLOC   ?= cloc

# Extra cloc flags (e.g. LOC_ARGS='--csv' make loc)
LOC_ARGS ?=

.PHONY: all install-dev install-python-deps install-system-mpi generate-artifacts legacy-build legacy-samplers legacy-clean test test-essential loc cloc example-mw example-sample example-halo-first clean help

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

clean: legacy-clean
	rm -rf $(VENV) build dist *.egg-info src/*.egg-info src/ntropy/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
