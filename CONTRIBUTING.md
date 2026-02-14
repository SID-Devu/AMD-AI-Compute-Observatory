<div align="center">

# 🤝 Contributing to AACO-SIGMA

**Thank you for your interest in contributing to the AMD AI Compute Observatory!**

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/pulls)
[![Contributors](https://img.shields.io/github/contributors/SID-Devu/AMD-AI-Compute-Observatory?style=flat-square)](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/graphs/contributors)

</div>

---

## 📋 Table of Contents

- [Development Setup](#-development-setup)
- [Code Style](#-code-style)
- [Testing](#-testing)
- [Pull Request Process](#-pull-request-process)
- [Architecture Guidelines](#-architecture-guidelines)

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.10+
- ROCm 6.0+ (for GPU features)
- Git

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/SID-Devu/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install in development mode with all dependencies
pip install -e ".[dev,full]"

# 4. Verify setup
pytest tests/unit/ -v --tb=short
```

---

## ✨ Code Style

We maintain high code quality standards using modern Python tooling.

### Linting & Formatting (Ruff)

```bash
# Check linting
ruff check aaco/

# Format code
ruff format aaco/

# Fix auto-fixable issues
ruff check --fix aaco/
```

### Type Checking (MyPy)

All code **must** include type hints:

```bash
mypy aaco/ --ignore-missing-imports
```

### Example Code Style

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
    """
    category: str
    confidence: float
    evidence: List[dict]
    recommendation: Optional[str] = None
```

---

## 🧪 Testing

### Test Categories

| Type | Location | Command | Requirements |
|------|----------|---------|--------------|
| **Unit** | `tests/unit/` | `pytest tests/unit -v` | None |
| **Integration** | `tests/integration/` | `pytest tests/integration -v` | ROCm |

### Running Tests

```bash
# Fast unit tests
pytest tests/unit -v

# Integration tests (requires ROCm)
AACO_RUN_INTEGRATION=1 pytest tests/integration -v

# Full suite with coverage
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

---

## 🔀 Pull Request Process

### 1️⃣ Fork & Branch

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory

# Create feature branch
git checkout -b feature/my-awesome-feature
```

### 2️⃣ Make Changes

- Follow code style guidelines
- Add tests for new functionality
- Update documentation if needed

### 3️⃣ Commit with Conventional Commits

```bash
git commit -m "feat: Add thermal throttle detection to classifier"
```

| Prefix | Description |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation |
| `test:` | Test changes |
| `refactor:` | Code refactoring |
| `perf:` | Performance |
| `chore:` | Maintenance |

### 4️⃣ Push & Create PR

```bash
git push origin feature/my-awesome-feature
```

Then create a Pull Request on GitHub.

---

## 🏗️ Architecture Guidelines

### Adding a New Collector

```
1. Create file: aaco/collectors/my_collector.py
2. Implement interface: start(), stop(), get_samples()
3. Export in: aaco/collectors/__init__.py
4. Document in: docs/data_schema.md
```

### Adding a New Bottleneck Category

```
1. Add to enum: BottleneckCategory in classify.py
2. Implement rules: BottleneckClassifier
3. Add recommendation: in recommendations mapping
4. Document: docs/bottleneck_taxonomy.md
```

### Adding a New CLI Command

```python
@cli.command()
@click.option('--model', required=True, help='Model name')
def my_command(model: str):
    """Brief description of command."""
    # Implementation
```

---

<div align="center">

## ❓ Questions?

Open an [issue](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/issues) or start a [discussion](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/discussions).

**Thank you for contributing to AACO-SIGMA! 🎉**

</div>
