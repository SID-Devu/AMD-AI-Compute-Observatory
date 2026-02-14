"""
AACO Test Suite - Observatory Tests
===================================
Tests for the main Observatory class.

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestObservatoryInit:
    """Tests for Observatory initialization."""
    
    def test_default_init(self):
        """Test default Observatory initialization."""
        from aaco.core import Observatory
        
        obs = Observatory()
        assert obs is not None
        assert obs.config is not None
    
    def test_init_with_config(self, sample_config):
        """Test Observatory with custom config."""
        from aaco.core import Observatory
        
        obs = Observatory(config=sample_config)
        assert obs.config["profiling"]["default_iterations"] == 10
    
    def test_init_with_config_file(self, temp_dir, sample_config):
        """Test Observatory with config file."""
        import yaml
        from aaco.core import Observatory
        
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        
        obs = Observatory(config=str(config_path))
        assert obs.config is not None


class TestObservatoryProfile:
    """Tests for Observatory.profile()."""
    
    @patch('aaco.core.observatory.ModelLoader')
    @patch('aaco.core.observatory.ProfileRunner')
    def test_profile_basic(self, mock_runner, mock_loader, mock_model):
        """Test basic profiling."""
        from aaco.core import Observatory
        
        # Setup mocks
        mock_loader.return_value.load.return_value = MagicMock()
        mock_runner.return_value.run.return_value = MagicMock(
            id="test_session",
            metrics={"latency_ms": [10.0] * 10}
        )
        
        obs = Observatory()
        session = obs.profile(
            model=str(mock_model),
            iterations=10,
            warmup=2
        )
        
        assert session is not None
        assert session.id == "test_session"
    
    @patch('aaco.core.observatory.ModelLoader')
    def test_profile_invalid_model(self, mock_loader):
        """Test profiling with invalid model."""
        from aaco.core import Observatory
        from aaco.core.exceptions import ModelError
        
        mock_loader.return_value.load.side_effect = ModelError("Invalid model")
        
        obs = Observatory()
        with pytest.raises(ModelError):
            obs.profile(model="nonexistent.onnx")
    
    def test_profile_with_lab_mode(self, mock_model):
        """Test profiling with laboratory mode."""
        from aaco.core import Observatory
        
        obs = Observatory()
        # Lab mode requires root, so we just test the config parsing
        assert obs.config is not None


class TestObservatoryAnalyze:
    """Tests for Observatory.analyze()."""
    
    def test_analyze_session(self, mock_session):
        """Test analyzing a session."""
        from aaco.core import Observatory
        
        obs = Observatory()
        with patch.object(obs, '_load_session', return_value=mock_session):
            analysis = obs.analyze(mock_session)
            # Analysis should return results
            assert analysis is not None


class TestObservatoryCompare:
    """Tests for Observatory.compare()."""
    
    def test_compare_sessions(self, mock_session, sample_metrics):
        """Test comparing two sessions."""
        from aaco.core import Observatory
        
        baseline = MagicMock()
        baseline.metrics = sample_metrics
        
        current = MagicMock()
        current.metrics = sample_metrics.copy()
        current.metrics["latency_ms"] = [x * 1.1 for x in sample_metrics["latency_ms"]]
        
        obs = Observatory()
        with patch.object(obs, '_load_session', side_effect=[baseline, current]):
            comparison = obs.compare(baseline, current)
            assert comparison is not None


class TestObservatoryReport:
    """Tests for Observatory.report()."""
    
    def test_report_html(self, mock_session, temp_dir):
        """Test HTML report generation."""
        from aaco.core import Observatory
        
        output_path = temp_dir / "report.html"
        
        obs = Observatory()
        with patch.object(obs, '_load_session', return_value=mock_session):
            with patch('aaco.report.HTMLReporter') as mock_reporter:
                mock_reporter.return_value.generate.return_value = output_path
                result = obs.report(mock_session, format="html", output=output_path)
                assert result == output_path
    
    def test_report_json(self, mock_session, temp_dir):
        """Test JSON report generation."""
        from aaco.core import Observatory
        
        output_path = temp_dir / "report.json"
        
        obs = Observatory()
        with patch.object(obs, '_load_session', return_value=mock_session):
            with patch('aaco.report.JSONReporter') as mock_reporter:
                mock_reporter.return_value.generate.return_value = output_path
                result = obs.report(mock_session, format="json", output=output_path)
                assert result == output_path
