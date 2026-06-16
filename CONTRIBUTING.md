# Contributing to Distill-Align

Thank you for your interest in contributing to Distill-Align! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry (recommended) or pip
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/distill-align.git
   cd distill-align
   ```

2. **Install dependencies**
   ```bash
   # Using Poetry (recommended)
   poetry install

   # Or using pip
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests**
   ```bash
   poetry run pytest
   ```

## Development Workflow

### Branch Naming

- `feature/your-feature` - New features
- `fix/your-bug-fix` - Bug fixes
- `docs/your-docs-update` - Documentation changes
- `refactor/your-refactor` - Code refactoring

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

### Code Style

We use Ruff for linting and formatting:

```bash
# Check code style
poetry run ruff check .

# Format code
poetry run ruff format .
```

### Type Checking

We use MyPy for static type checking:

```bash
poetry run mypy src/
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run checks**
   ```bash
   poetry run pytest
   poetry run ruff check .
   poetry run mypy src/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat(module): description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature
   ```

6. **Create a Pull Request**
   - Provide a clear description
   - Reference related issues
   - Include test results

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/logs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
