# Contributing

Thank you for your interest in contributing to Distill-Align! This guide covers the development workflow, code style, and testing practices.

> **Note**: Also see our [Contributor Covenant Code of Conduct](https://github.com/omargargoum/Distill-Align/blob/main/CODE_OF_CONDUCT.md).

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/) (recommended)
- Git
- (Optional) An LLM provider account (OpenAI, Ollama, etc.)

### Clone and Install

```bash
git clone https://github.com/omargargoum/Distill-Align.git
cd Distill-Align
poetry install
```

### Activate the Virtual Environment

```bash
poetry shell
```

Or run commands directly:

```bash
poetry run distill-align --help
```

## Code Style

We use **Ruff** for both linting and formatting:

```bash
# Format code
poetry run ruff format .

# Check lint
poetry run ruff check .

# Auto-fix issues
poetry run ruff check --fix .
```

Configuration is in `pyproject.toml`:
- Line length: 120
- Quote style: double quotes
- Target: Python 3.11

### Type Checking

We use **MyPy** for static type checking:

```bash
poetry run mypy src/
```

Run both checks before committing:

```bash
poetry run ruff check . && poetry run mypy src/
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/

# Run specific test file
poetry run pytest tests/test_ingestion.py -v
```

### Writing Tests

- Place tests in the `tests/` directory, mirroring the `src/` structure
- Use descriptive test function names: `test_ingest_markdown_file`
- Use fixtures for shared test data
- Prefer async tests with `pytest-asyncio` for pipeline tests
- Mock external services (LLM APIs) in unit tests

## Pre-commit Hooks

We use pre-commit hooks to enforce code quality:

```bash
# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Pull Request Workflow

### Before Submitting

1. Fork the repository and create a feature branch
2. Make your changes with clear, documented code
3. Add tests for new functionality
4. Update documentation as needed
5. Run all checks: `pytest`, `ruff`, `mypy`
6. Commit using conventional commit format

### Conventional Commits

```
feat(ingestion): add support for EPUB files
fix(synthesis): handle rate limit errors gracefully
docs(cli): update export command examples
```

### PR Checklist

- [ ] Code follows project style (ruff format)
- [ ] Self-review performed
- [ ] Tests added/passed
- [ ] Documentation updated
- [ ] No new warnings or errors

## Project Structure

```
src/distill_align/
├── cli/         # Typer CLI entry points
├── core/        # Config, schemas, logging, exceptions
├── ingestion/   # File loaders and chunkers
├── synthesis/   # LLM clients, prompts, worker pool
├── exporter/    # Formatters, splitter, validator
└── tui/         # Textual terminal UI
```

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and ideas
- **Pull Requests**: Code contributions
