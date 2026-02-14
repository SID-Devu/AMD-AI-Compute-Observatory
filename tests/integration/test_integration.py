"""
AACO Integration Tests
Tests requiring full system components.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Skip if ONNX Runtime not available
pytest.importorskip("onnxruntime")


class TestORTRunnerIntegration:
    """Integration tests for ONNX Runtime runner."""
    
    @pytest.fixture
    def simple_model_path(self, tmp_path):
        """Create a simple ONNX model for testing."""
        import onnx
        from onnx import helper, TensorProto
        
        # Create a simple Add model
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 10])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])
        Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 10])
        
        add_node = helper.make_node('Add', ['X', 'Y'], ['Z'])
        
        graph = helper.make_graph([add_node], 'test_add', [X, Y], [Z])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        
        model_path = tmp_path / "test_model.onnx"
        onnx.save(model, str(model_path))
        
        return model_path
    
    @pytest.mark.skipif(
        not os.environ.get("AACO_RUN_INTEGRATION"),
        reason="Integration tests disabled (set AACO_RUN_INTEGRATION=1)"
    )
    def test_ort_runner_cpu(self, simple_model_path):
        """Test ORT runner with CPU backend."""
        from aaco.runner.ort_runner import ORTRunner, RunConfig
        
        config = RunConfig(
            backend="cpu",
            warmup_iterations=2,
            measurement_iterations=5,
        )
        
        runner = ORTRunner(simple_model_path, config=config)
        results = runner.run_benchmark()
        
        assert len(results) == 7  # 2 warmup + 5 measurement
        
        warmup_results = [r for r in results if r.phase == "warmup"]
        measure_results = [r for r in results if r.phase == "measurement"]
        
        assert len(warmup_results) == 2
        assert len(measure_results) == 5
        
        # All latencies should be positive
        for r in results:
            assert r.latency_ms > 0


class TestSessionIntegration:
    """Integration tests for session management."""
    
    def test_full_session_workflow(self):
        """Test complete session creation and artifact storage."""
        from aaco.core.session import SessionManager, Session
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(base_dir=Path(tmpdir))
            
            # Create session
            session = manager.create_session(model_name="test_model", backend="cpu")
            session.initialize()  # Creates session.json and metadata
            
            # Save various artifacts
            session.save_artifact("misc", "test_data.json", {"key": "value", "number": 42})
            session.save_artifact("misc", "list_data.json", [1, 2, 3, 4, 5])
            
            # Verify artifacts exist
            assert (session.session_dir / "misc" / "test_data.json").exists()
            assert (session.session_dir / "misc" / "list_data.json").exists()
            
            # Verify content
            with open(session.session_dir / "misc" / "test_data.json") as f:
                data = json.load(f)
                assert data["key"] == "value"
            
            # Test session listing
            sessions = manager.list_sessions()
            assert len(sessions) >= 1


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_cli_info_command(self):
        """Test CLI info command."""
        from click.testing import CliRunner
        from aaco.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['info'])
        
        assert result.exit_code == 0
        assert "Python" in result.output
    
    def test_cli_ls_command(self):
        """Test CLI ls command."""
        from click.testing import CliRunner
        from aaco.cli import cli
        
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['ls', '--output', './sessions'])
            
            # Should not fail even with empty directory
            assert result.exit_code == 0


class TestReportIntegration:
    """Integration tests for report generation."""
    
    def test_report_from_session(self):
        """Test report generation from session data."""
        from aaco.report.render import ReportRenderer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "test_session"
            session_path.mkdir()
            
            # Create mock session data
            metadata = {
                "session_id": "test_123",
                "timestamp": "2024-01-15T10:00:00",
                "hostname": "testhost",
            }
            
            metrics = {
                "measurement_mean_ms": 10.5,
                "measurement_std_ms": 1.2,
                "measurement_p99_ms": 13.0,
                "throughput_ips": 95.2,
            }
            
            bottleneck = {
                "primary": "balanced",
                "secondary": [],
                "confidence": 0.8,
                "evidence": ["Balanced performance"],
                "recommendations": ["No critical issues"],
            }
            
            # Write mock data
            (session_path / "session_meta.json").write_text(json.dumps(metadata))
            (session_path / "derived_metrics.json").write_text(json.dumps(metrics))
            (session_path / "bottleneck.json").write_text(json.dumps(bottleneck))
            
            # Generate report
            renderer = ReportRenderer(session_path)
            
            # Test terminal report
            terminal_output = renderer.render_terminal()
            assert "10.5" in terminal_output  # Mean latency
            assert "balanced" in terminal_output.lower()
            
            # Test JSON report
            json_output = renderer.render_json()
            data = json.loads(json_output)
            assert "metadata" in data
            assert "metrics" in data
            
            # Test HTML report
            html_output = renderer.render_html()
            assert "<html" in html_output
            assert "10.5" in html_output
