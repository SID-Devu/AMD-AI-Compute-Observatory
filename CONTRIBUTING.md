# Contributing to AMD AI Compute Observatory

Thank you for your interest in contributing to AMD AI Compute Observatory (AACO). This document provides guidelines and instructions for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Architecture Guidelines](#architecture-guidelines)

---

## Code of Conduct

Contributors are expected to maintain professional conduct in all interactions. Be respectful, constructive, and focused on technical merit.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- ROCm 6.0+ (for integration testing)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/SID-Devu/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev,all]"

# Verify installation
pytest tests/unit/ -v --tb=short
```

---

## Development Environment

### Required Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| Ruff | Linting and formatting | `pip install ruff` |
| MyPy | Static type checking | `pip install mypy` |
| pytest | Testing framework | `pip install pytest` |
| pre-commit | Git hooks | `pip install pre-commit` |

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

---

## Code Style

AACO follows strict code quality standards. All submissions must pass automated checks.

### Linting and Formatting

```bash
# Check for issues
ruff check aaco/

# Auto-fix issues
ruff check --fix aaco/

# Format code
ruff format aaco/
```

### Type Annotations

All code must include type hints:

```bash
mypy aaco/ --ignore-missing-imports
```

### Documentation

- Use Google-style docstrings
- Document all public functions, classes, and methods
- Include type information in docstrings

**Example:**

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class BottleneckResult:
    """Result of bottleneck classification analysis.
    
    Attributes:
        category: The detected bottleneck category.
        confidence: Confidence score between 0 and 1.
        evidence: List of evidence signals supporting classification.
        recommendation: Optional optimization recommendation.
    """
    category: str
    confidence: float
    evidence: List[dict]
    recommendation: Optional[str] = None
```

---

## Testing

### Test Structure

| Type | Location | Purpose |
|------|----------|---------|
| Unit | `tests/unit/` | Component-level testing |
| Integration | `tests/integration/` | End-to-end testing (requires ROCm) |

### Running Tests

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# With coverage
pytest --cov=aaco --cov-report=html --cov-report=term
```

### Writing Tests

```python
import pytest
from aaco.analytics.classify import BottleneckClassifier

class TestBottleneckClassifier:
    """Tests for BottleneckClassifier."""
    
    def test_launch_bound_detection(self):
        """Should detect launch-bound workloads correctly."""
        classifier = BottleneckClassifier()
        metrics = {"microkernel_pct": 0.8, "launch_rate": 15000}
        
        result = classifier.classify(metrics)
        
        assert result.category == "launch-bound"
        assert result.confidence > 0.7
    
    def test_invalid_input_raises(self):
        """Should raise ValueError for invalid input."""
        classifier = BottleneckClassifier()
        
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            classifier.classify({})
```

### Test Requirements

- All new features must include tests
- Maintain minimum 80% code coverage
- Tests must pass on all supported platforms

---

## Pull Request Process

### 1. Create a Branch

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory

# Create feature branch
git checkout -b feature/descriptive-name
```

### 2. Implement Changes

- Follow code style guidelines
- Add or update tests as needed
- Update documentation if applicable

### 3. Commit Changes

Use conventional commit messages:

| Prefix | Description |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `test:` | Test additions or modifications |
| `refactor:` | Code refactoring |
| `perf:` | Performance improvements |
| `chore:` | Maintenance tasks |

**Example:**

```bash
git commit -m "feat: add thermal throttle detection to classifier"
```

### 4. Submit Pull Request

```bash
git push origin feature/descriptive-name
```

Create a pull request on GitHub with:

- Clear title describing the change
- Description of what was changed and why
- Reference to any related issues
- Test results and coverage information

### 5. Review Process

- All PRs require review before merging
- Address review feedback promptly
- CI checks must pass

---

## Architecture Guidelines

### Adding a New Collector

1. Create file: `aaco/collectors/my_collector.py`
2. Implement interface: `start()`, `stop()`, `get_samples()`
3. Export in: `aaco/collectors/__init__.py`
4. Add tests: `tests/unit/collectors/test_my_collector.py`
5. Document in: `docs/data_schema.md`

### Adding a New Bottleneck Category

1. Add to enum: `BottleneckCategory` in `classify.py`
2. Implement detection rules in `BottleneckClassifier`
3. Add recommendation mapping
4. Update documentation: `docs/bottleneck_taxonomy.md`
5. Add tests for the new category

### Adding a CLI Command

```python
@cli.command()
@click.option('--model', required=True, help='Model name or path')
def my_command(model: str) -> None:
    """Brief description of what the command does."""
    # Implementation
```

---

## Questions

For questions or clarification:

- Open an [issue](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/issues)
- Start a [discussion](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/discussions)

---

## License

By contributing to AMD AI Compute Observatory, you agree that your contributions will be licensed under the MIT License.
