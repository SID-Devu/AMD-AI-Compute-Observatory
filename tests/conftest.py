"""
AACO Test Configuration and Fixtures
=====================================
Shared fixtures and configuration for all tests.

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    path = Path(tempfile.mkdtemp(prefix="aaco_test_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration dictionary."""
    return {
        "profiling": {
            "default_iterations": 10,
            "default_warmup": 2,
        },
        "laboratory": {
            "enabled": False,
        },
        "analysis": {
            "statistics": {
                "confidence_level": 0.95,
            }
        }
    }


@pytest.fixture
def sample_metrics() -> dict:
    """Sample metrics data."""
    import numpy as np
    np.random.seed(42)
    
    return {
        "latency_ms": np.random.normal(10.0, 0.5, 100).tolist(),
        "throughput_fps": 100.0,
        "gpu_utilization": 85.5,
        "memory_used_mb": 1024,
        "power_watts": 150.0,
    }


@pytest.fixture
def mock_gpu():
    """Mock GPU device."""
    gpu = MagicMock()
    gpu.device_id = 0
    gpu.name = "AMD Instinct MI200"
    gpu.memory_total = 64 * 1024  # 64 GB
    gpu.compute_units = 220
    gpu.clock_speed = 1700  # MHz
    return gpu


@pytest.fixture
def mock_session(temp_dir, sample_metrics):
    """Create a mock profiling session."""
    from aaco.core import Session
    
    session_dir = temp_dir / "session_test"
    session_dir.mkdir(parents=True)
    
    # Create mock session
    session = MagicMock(spec=Session)
    session.id = "test_session_001"
    session.path = session_dir
    session.metrics = sample_metrics
    session.config = {"iterations": 100, "warmup": 10}
    
    return session


@pytest.fixture
def mock_model(temp_dir):
    """Create a mock ONNX model file."""
    model_path = temp_dir / "test_model.onnx"
    # Create a minimal valid ONNX file (header only for testing)
    model_path.write_bytes(b"ONNX" + b"\x00" * 100)
    return model_path


# =============================================================================
# Test Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "rocm: marks tests requiring ROCm")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")


# =============================================================================
# Skip Conditions
# =============================================================================

def has_rocm() -> bool:
    """Check if ROCm is available."""
    try:
        from pathlib import Path
        return Path("/opt/rocm").exists()
    except Exception:
        return False


def has_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


skip_no_rocm = pytest.mark.skipif(
    not has_rocm(),
    reason="ROCm not installed"
)

skip_no_gpu = pytest.mark.skipif(
    not has_gpu(),
    reason="No GPU available"
)
