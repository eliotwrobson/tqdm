# Contributing to TL;DM

Thank you for your interest in contributing to TL;DM! This guide will help you get started with the development workflow.

## Code of Conduct

When interacting with other users and maintainers, please be sure to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Submitting an Issue

### Bug Reports

If you are submitting a bug report, please answer the following questions:

1. What version of TL;DM were you using?
2. What were you doing?
3. What did you expect to happen?
4. What happened instead?

Please provide any code to reproduce the issue, if possible.

### Feature Requests

If you are requesting a new feature or change in behavior, please describe what you are looking for, and what value it will add to your use case.

## Modifying the Codebase

TL;DM is an open-source project, so you are welcome and encouraged to modify the codebase with new fixes and enhancements. Please observe the following guidelines when submitting pull requests:

### Setting Up Your Environment

The dependencies for the project are managed with [uv](https://github.com/astral-sh/uv). For instructions on how to install `uv`, see the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

To create a virtual environment and install all project dependencies, run:

```bash
uv sync
```

### Code Quality Standards

1. **Code Formatting and Linting**: All code must comply with the enabled ruff lint rules. If you have installed the development dependencies, the `ruff` package will be available with the config in `pyproject.toml`. Run the formatter with:

```bash
make format
# or: uv run ruff format src/tldm tests
```

2. **Type Checking**: All new code must include type annotations and pass type checking with [mypy](https://mypy.readthedocs.io/en/stable/). Run type checking with:

```bash
make mypy
# or: uv run mypy src/tldm/std.py src/tldm/utils.py src/tldm/aliases.py
```

3. **Testing**: Whether you are introducing a bug fix or a new feature, you must add tests to verify that your code additions function correctly. Run tests with:

```bash
make pytest
# or: uv run pytest
```

4. **Code Coverage**: Please ensure your changes are covered by tests. Run tests with coverage:

```bash
make pytest-cov
# or: uv run pytest --cov=tldm --cov-report=xml --cov-report=term --cov-report=html
```

If the coverage report needs detailed review, you can open the HTML coverage report:

```bash
open htmlcov/index.html  # macOS/Linux
start htmlcov\index.html  # Windows
```

5. **Documentation**: If you are adding a new feature or changing behavior, please update the documentation appropriately. This includes updating docstrings for all functions in the public interface, using the [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).

### Running All Checks

You can run all formatting, linting, and tests with a single command:

```bash
make all
```

This will:

1. Format code with ruff
2. Run linting checks
3. Run mypy type checking
4. Run pytest with coverage

### Available Make Commands

The project includes a Makefile with convenient shortcuts. Run `make help` to see all available commands:

- `make format` - Format code with ruff
- `make lint` - Run ruff linting and mypy type checking
- `make mypy` - Run mypy type checking only
- `make pytest` - Run pytest without coverage
- `make pytest-cov` - Run pytest with coverage reports
- `make all` - Run formatting, linting, and tests with coverage
- `make clean` - Clean up build artifacts and cache files

## Pull Request Workflow

Contributions to the project are made using the "Fork & Pull" model:

1. Fork the [`tldm`](https://github.com/eliotwrobson/tldm) repository
2. Clone your fork locally: `git clone https://github.com/your_account/tldm.git`
3. Create a new branch for your changes: `git checkout -b my-feature-branch`
4. Make your changes on the local copy
5. Ensure all tests pass and code is formatted: `make all`
6. Commit your changes: `git commit -m "Description of changes"`
7. Push to your GitHub fork: `git push origin my-feature-branch`
8. Create a Pull Request from your GitHub fork (go to your fork's webpage and click "Pull Request")

### Before Submitting a Pull Request

Please ensure:

- All tests pass locally
- Code is properly formatted (run `make format`)
- Type checking passes (run `make mypy`)
- Your changes include appropriate tests
- Documentation is updated if needed

## Questions or Issues?

If you have questions or run into issues while contributing, feel free to:

- Open an issue on [GitHub](https://github.com/eliotwrobson/tldm/issues)
- Check existing issues and pull requests for similar discussions

Thank you for contributing to TL;DM!
