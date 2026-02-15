# Contributing to AMD AI Compute Observatory

Thank you for your interest in contributing to AMD AI Compute Observatory. This document outlines
the contribution process and requirements for the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

All contributors are expected to adhere to professional standards of conduct. Please review our
[Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Required |
| Git | 2.30+ | Required |
| ROCm | 6.0+ | Required for integration tests |

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/<your-username>/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/SID-Devu/AMD-AI-Compute-Observatory.git
```

## Development Setup

### Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### Install Dependencies

```bash
pip install -e ".[dev,all]"
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Coding Standards

### Style Guidelines

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. All
submissions must pass automated style checks.

```bash
# Check for style issues
ruff check aaco/

# Auto-fix issues where possible
ruff check --fix aaco/

# Format code
ruff format aaco/
```

### Type Annotations

All code must include type hints. Type checking is performed with MyPy:

```bash
mypy aaco/ --ignore-missing-imports
```

### Documentation Standards

- All public functions, classes, and modules must have docstrings
- Use Google-style docstring format
- Include type information in docstrings for complex types
- Update relevant documentation when adding or modifying features

Example:

```python
def compute_heu(measured: float, calibrated: float) -> float:
    """Compute Hardware Envelope Utilization.

    Args:
        measured: Measured throughput in operations per second.
        calibrated: Calibrated peak throughput from hardware envelope.

    Returns:
        HEU as a decimal between 0.0 and 1.0.

    Raises:
        ValueError: If calibrated value is zero or negative.
    """
    if calibrated <= 0:
        raise ValueError("Calibrated value must be positive")
    return measured / calibrated
```

## Testing Requirements

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires ROCm)
pytest tests/integration/ -v

# Full suite with coverage
pytest --cov=aaco --cov-report=html
```

### Test Coverage

- New features must include corresponding unit tests
- Bug fixes should include regression tests
- Maintain minimum 80% code coverage for new code

### Test Organization

| Directory | Purpose |
|-----------|---------|
| `tests/unit/` | Unit tests for individual components |
| `tests/integration/` | Integration tests requiring full environment |
| `tests/benchmarks/` | Performance benchmarks |

## Submitting Changes

### Branch Naming

Use descriptive branch names following the pattern:

- `feature/<description>` — New features
- `fix/<description>` — Bug fixes
- `docs/<description>` — Documentation updates
- `refactor/<description>` — Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:

```
feat(analytics): add Bayesian attribution engine

fix(profiler): correct kernel duration calculation for async launches

docs(readme): update installation instructions for ROCm 6.2
```

### Pull Request Process

1. Ensure all tests pass locally
2. Update documentation as needed
3. Create pull request against the `master` branch
4. Fill out the pull request template completely
5. Request review from maintainers

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New code includes appropriate tests
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] No merge conflicts

## Review Process

### Timeline

- Initial review within 5 business days
- Follow-up reviews within 3 business days

### Review Criteria

- Code quality and style compliance
- Test coverage and quality
- Documentation completeness
- Performance implications
- Security considerations

### Merging

Pull requests require:

- At least one approving review
- All CI checks passing
- No unresolved comments
- Up-to-date with target branch

## Questions

For questions about contributing, please open a
[Discussion](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/discussions) on GitHub.
