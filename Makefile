.PHONY: help install test lint format type-check clean build publish docs

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	poetry install

install-dev: ## Install with dev dependencies
	poetry install --with dev

test: ## Run tests
	poetry run pytest

test-cov: ## Run tests with coverage
	poetry run pytest --cov=src/distill_align --cov-report=html

lint: ## Run linter
	poetry run ruff check .

format: ## Format code
	poetry run ruff format .

type-check: ## Run type checking
	poetry run mypy src/

clean: ## Clean build artifacts
	poetry run python -c "
import shutil, pathlib, os, sys
dirs = ['build', 'dist', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov']
files = ['.coverage']
for d in dirs:
    p = pathlib.Path(d)
    if p.exists():
        shutil.rmtree(p)
        print(f'Removed {d}')
for f in files:
    p = pathlib.Path(f)
    if p.exists():
        p.unlink()
        print(f'Removed {f}')
for d in pathlib.Path('.').rglob('__pycache__'):
    shutil.rmtree(d)
    print(f'Removed {d}')
for p in pathlib.Path('.').rglob('*.egg-info'):
    shutil.rmtree(p)
    print(f'Removed {p}')
for p in pathlib.Path('.').rglob('*.egg'):
    p.unlink()
    print(f'Removed {p}')
"

build: ## Build package
	poetry build

publish: ## Publish to PyPI
	poetry publish

docs: ## Build documentation
	poetry run mkdocs build

docs-serve: ## Serve documentation locally
	poetry run mkdocs serve

pre-commit: ## Run pre-commit hooks
	poetry run pre-commit run --all-files

check: lint type-check test ## Run all checks

.DEFAULT_GOAL := help
