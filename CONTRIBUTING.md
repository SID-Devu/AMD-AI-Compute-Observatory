# Contributing to AACO

Thank you for your interest in contributing to the AMD AI Compute Observatory (AACO)!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sudheerdevu/AMD-AI-Compute-Observatory.git
   cd AMD-AI-Compute-Observatory
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev,full]"
   ```

4. Run tests:
   ```bash
   pytest tests/unit/ -v
   ```

## Code Style

We use **Ruff** for linting and formatting:

```bash
# Check linting
ruff check aaco/

# Format code
ruff format aaco/

# Fix auto-fixable issues
ruff check --fix aaco/
```

Configuration is in `pyproject.toml`.

## Type Hints

All code should include type hints. We use **MyPy** for type checking:

```bash
mypy aaco/ --ignore-missing-imports
```

## Testing

### Unit Tests
- Located in `tests/unit/`
- Run with: `pytest tests/unit/ -v`
- Should not require external dependencies (mock when needed)

### Integration Tests
- Located in `tests/integration/`
- May require ONNX Runtime, rocprof, rocm-smi
- Set `AACO_RUN_INTEGRATION=1` to enable

### Writing Tests
```python
import pytest
from aaco.core.session import Session

class TestNewFeature:
    def test_basic_functionality(self):
        """Test description."""
        result = my_function()
        assert result == expected
    
    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Make changes** following our code style
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run tests** to ensure everything passes
7. **Commit** with clear messages:
   ```bash
   git commit -m "feat: Add new bottleneck classifier for data transfer"
   ```
8. **Push** to your fork and create a **Pull Request**

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

Examples:
```
feat: Add thermal throttle detection to bottleneck classifier
fix: Handle empty rocprof CSV files gracefully
docs: Update architecture diagram with new analytics module
test: Add unit tests for regression detector
```

## Architecture Guidelines

### Adding a New Collector
1. Create file in `aaco/collectors/`
2. Implement `start()`, `stop()`, `get_samples()` interface
3. Add to `__init__.py` exports
4. Document sample schema in `docs/data_schema.md`

### Adding a New Bottleneck Category
1. Add to `BottleneckCategory` enum in `classify.py`
2. Implement detection rules in `BottleneckClassifier`
3. Add recommendations mapping
4. Document in `docs/bottleneck_taxonomy.md`

### Adding a New CLI Command
1. Add command function in `cli.py` with `@cli.command()` decorator
2. Use Click options/arguments for parameters
3. Add help text and examples
4. Test with Click's `CliRunner`

## Documentation

- **Code Comments**: Explain *why*, not *what*
- **Docstrings**: Use Google-style docstrings
- **README**: Keep examples up to date
- **Architecture**: Update `docs/architecture.md` for structural changes

## Questions?

Open an issue for questions or discussion about contributions.
