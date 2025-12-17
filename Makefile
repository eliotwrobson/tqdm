.PHONY: help mypy ruff ruff-format ruff-check format lint pytest pytest-cov test all install install-dev clean

help:
	@echo "Available commands:"
	@echo "  make mypy          - Run mypy type checking on core modules"
	@echo "  make ruff-format   - Format code with ruff"
	@echo "  make ruff-check    - Lint code with ruff"
	@echo "  make ruff          - Format and lint with ruff"
	@echo "  make format        - Alias for ruff-format"
	@echo "  make lint          - Run ruff linting and mypy type checking"
	@echo "  make pytest        - Run pytest without coverage"
	@echo "  make pytest-cov    - Run pytest with coverage reports"
	@echo "  make test          - Alias for pytest"
	@echo "  make all           - Run formatting, linting, and tests with coverage"
	@echo "  make install       - Install the package"
	@echo "  make install-dev   - Install the package with development dependencies"
	@echo "  make clean         - Clean up build artifacts and cache files"

mypy:
	@echo "Running mypy type checking..."
	uv run mypy src/tldm/std.py src/tldm/utils.py src/tldm/aliases.py

ruff-format:
	@echo "Running ruff formatter..."
	uv run ruff format src/tldm tests

ruff-check:
	@echo "Running ruff linter..."
	uv run ruff check src/tldm tests
	@echo "Checking ruff formatting..."
	uv run ruff format --check src/tldm tests

ruff: ruff-format ruff-check

format: ruff-format

lint: ruff-check mypy

pytest:
	@echo "Running pytest without coverage..."
	uv run pytest

pytest-cov:
	@echo "Running pytest with coverage..."
	uv run pytest --cov=tldm --cov-report=xml --cov-report=term --cov-report=html

test: pytest

all: format lint pytest-cov
	@echo "All checks completed successfully!"

install:
	uv pip install .

install-dev:
	uv pip install -e ".[dev]"

clean:
	@python -c "import shutil; shutil.rmtree('build', True); shutil.rmtree('dist', True); shutil.rmtree('.eggs', True)"
	@python -c "import os, glob; [os.remove(i) for i in glob.glob('.coverage*')]"
	@python -c "import shutil; shutil.rmtree('.pytest_cache', True); shutil.rmtree('.mypy_cache', True); shutil.rmtree('.ruff_cache', True)"
	@python -c "import shutil; shutil.rmtree('htmlcov', True)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.py[co]' -delete
	@echo "Clean completed!"
