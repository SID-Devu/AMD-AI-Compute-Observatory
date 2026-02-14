"""
AACO Test Suite - CLI Tests
===========================
Tests for command-line interface.

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock


@pytest.fixture
def cli_runner():
    """Create CLI test runner."""
    return CliRunner()


class TestCLIHelp:
    """Tests for CLI help commands."""
    
    def test_main_help(self, cli_runner):
        """Test main help output."""
        from aaco.cli import main
        
        result = cli_runner.invoke(main, ["--help"])
        
        assert result.exit_code == 0
        assert "AACO" in result.output or "aaco" in result.output
    
    def test_profile_help(self, cli_runner):
        """Test profile command help."""
        from aaco.cli import main
        
        result = cli_runner.invoke(main, ["profile", "--help"])
        
        assert result.exit_code == 0
        assert "--model" in result.output
    
    def test_analyze_help(self, cli_runner):
        """Test analyze command help."""
        from aaco.cli import main
        
        result = cli_runner.invoke(main, ["analyze", "--help"])
        
        assert result.exit_code == 0
        assert "--session" in result.output
    
    def test_report_help(self, cli_runner):
        """Test report command help."""
        from aaco.cli import main
        
        result = cli_runner.invoke(main, ["report", "--help"])
        
        assert result.exit_code == 0
        assert "--format" in result.output


class TestCLIVersion:
    """Tests for version commands."""
    
    def test_version_flag(self, cli_runner):
        """Test --version flag."""
        from aaco.cli import main
        
        result = cli_runner.invoke(main, ["--version"])
        
        assert result.exit_code == 0
        # Should contain version number
        assert "." in result.output  # e.g., "1.0.0"


class TestCLIProfile:
    """Tests for profile command."""
    
    @patch('aaco.cli.Observatory')
    def test_profile_basic(self, mock_obs_class, cli_runner, tmp_path):
        """Test basic profile command."""
        from aaco.cli import main
        
        # Create mock model
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"ONNX")
        
        # Setup mock
        mock_obs = MagicMock()
        mock_obs.profile.return_value = MagicMock(id="test_session")
        mock_obs_class.return_value = mock_obs
        
        result = cli_runner.invoke(main, [
            "profile",
            "--model", str(model_path),
            "--output", str(tmp_path / "output")
        ])
        
        # Should call profile
        assert mock_obs.profile.called
    
    def test_profile_missing_model(self, cli_runner):
        """Test profile with missing model."""
        from aaco.cli import main
        
        result = cli_runner.invoke(main, [
            "profile",
            "--model", "nonexistent.onnx"
        ])
        
        # Should fail
        assert result.exit_code != 0


class TestCLIAnalyze:
    """Tests for analyze command."""
    
    @patch('aaco.cli.Observatory')
    def test_analyze_basic(self, mock_obs_class, cli_runner, tmp_path):
        """Test basic analyze command."""
        from aaco.cli import main
        
        # Create mock session directory
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        (session_dir / "session.json").write_text("{}")
        
        # Setup mock
        mock_obs = MagicMock()
        mock_obs.analyze.return_value = MagicMock()
        mock_obs_class.return_value = mock_obs
        
        result = cli_runner.invoke(main, [
            "analyze",
            "--session", str(session_dir)
        ])
        
        assert mock_obs.analyze.called


class TestCLIReport:
    """Tests for report command."""
    
    @patch('aaco.cli.Observatory')
    def test_report_html(self, mock_obs_class, cli_runner, tmp_path):
        """Test HTML report generation."""
        from aaco.cli import main
        
        # Create mock session
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        (session_dir / "session.json").write_text("{}")
        
        output_path = tmp_path / "report.html"
        
        # Setup mock
        mock_obs = MagicMock()
        mock_obs.report.return_value = output_path
        mock_obs_class.return_value = mock_obs
        
        result = cli_runner.invoke(main, [
            "report",
            "--session", str(session_dir),
            "--format", "html",
            "--output", str(output_path)
        ])
        
        assert mock_obs.report.called
    
    @patch('aaco.cli.Observatory')
    def test_report_json(self, mock_obs_class, cli_runner, tmp_path):
        """Test JSON report generation."""
        from aaco.cli import main
        
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        (session_dir / "session.json").write_text("{}")
        
        output_path = tmp_path / "report.json"
        
        mock_obs = MagicMock()
        mock_obs.report.return_value = output_path
        mock_obs_class.return_value = mock_obs
        
        result = cli_runner.invoke(main, [
            "report",
            "--session", str(session_dir),
            "--format", "json",
            "--output", str(output_path)
        ])
        
        assert mock_obs.report.called


class TestCLIDoctor:
    """Tests for doctor command."""
    
    def test_doctor_basic(self, cli_runner):
        """Test basic doctor command."""
        from aaco.cli import main
        
        result = cli_runner.invoke(main, ["doctor"])
        
        # Should run without error
        assert result.exit_code in [0, 1]  # 0 = ok, 1 = issues found
