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
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

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
