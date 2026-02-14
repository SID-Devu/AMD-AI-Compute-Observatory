"""
AACO Test Suite - CLI Tests
===========================
Tests for command-line interface.

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import pytest
from click.testing import CliRunner

from aaco.cli import cli


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


class TestCLIHelp:
    """Tests for CLI help commands."""
    
    def test_main_help(self, runner):
        """Test main help output."""
        result = runner.invoke(cli, ["--help"])
        
        assert result.exit_code == 0
        assert "AACO" in result.output or "aaco" in result.output
    
    def test_run_help(self, runner):
        """Test run command help."""
        result = runner.invoke(cli, ["run", "--help"])
        
        assert result.exit_code == 0
        assert "model" in result.output.lower()
    
    def test_report_help(self, runner):
        """Test report command help."""
        result = runner.invoke(cli, ["report", "--help"])
        
        assert result.exit_code == 0
    
    def test_diff_help(self, runner):
        """Test diff command help."""
        result = runner.invoke(cli, ["diff", "--help"])
        
        assert result.exit_code == 0


class TestCLIVerbose:
    """Test verbose flag."""
    
    def test_verbose_flag(self, runner):
        """Test --verbose flag is recognized."""
        result = runner.invoke(cli, ["--verbose", "--help"])
        
        assert result.exit_code == 0
