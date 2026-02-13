"""
rocprof Wrapper
Launches workloads under rocprof profiling and manages output files.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from aaco.core.utils import run_command, ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class RocprofConfig:
    """Configuration for rocprof profiling."""
    output_dir: Path
    trace_hip: bool = True
    trace_hsa: bool = False
    trace_sys: bool = False
    hip_api: bool = True
    hsa_api: bool = False
    kernel_trace: bool = True
    stats: bool = True
    basenames: bool = True
    timestamp: str = "on"
    flush_interval: int = 0
    extra_args: List[str] = None
    
    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = []


class RocprofWrapper:
    """
    Wrapper around rocprof for GPU kernel profiling.
    Handles command construction, execution, and output management.
    """
    
    def __init__(self, config: RocprofConfig):
        self.config = config
        self._available = self._check_rocprof()
        self._output_files: List[Path] = []
    
    def _check_rocprof(self) -> bool:
        """Check if rocprof is available."""
        result = run_command(["rocprof", "--version"])
        if result:
            logger.info(f"rocprof available: {result.strip().split(chr(10))[0]}")
            return True
        return False
    
    @property
    def available(self) -> bool:
        """Check if rocprof is available on the system."""
        return self._available
    
    def build_command(self, workload_cmd: List[str]) -> List[str]:
        """Build the full rocprof command with all options."""
        cmd = ["rocprof"]
        
        # Output directory
        ensure_dir(self.config.output_dir)
        
        # Basic options
        if self.config.stats:
            cmd.append("--stats")
        
        if self.config.basenames:
            cmd.append("--basenames")
            cmd.append("on")
        
        if self.config.timestamp:
            cmd.extend(["--timestamp", self.config.timestamp])
        
        # Tracing options
        if self.config.trace_hip:
            cmd.append("--hip-trace")
        
        if self.config.trace_hsa:
            cmd.append("--hsa-trace")
        
        if self.config.trace_sys:
            cmd.append("--sys-trace")
        
        # API tracing
        if self.config.hip_api:
            cmd.append("--hip-api")
        
        if self.config.hsa_api:
            cmd.append("--hsa-api")
        
        # Output file prefix
        output_prefix = self.config.output_dir / "rocprof"
        cmd.extend(["-o", str(output_prefix)])
        
        # Flush interval
        if self.config.flush_interval > 0:
            cmd.extend(["--flush-interval", str(self.config.flush_interval)])
        
        # Extra arguments
        cmd.extend(self.config.extra_args)
        
        # Add separator and workload command
        cmd.append("--")
        cmd.extend(workload_cmd)
        
        return cmd
    
    def run(
        self,
        workload_cmd: List[str],
        env: Optional[Dict[str, str]] = None,
        timeout: int = 600,
    ) -> Tuple[bool, str, List[Path]]:
        """
        Run workload under rocprof profiling.
        
        Args:
            workload_cmd: Command to profile
            env: Environment variables
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, stdout/stderr, output_files)
        """
        if not self._available:
            logger.error("rocprof not available")
            return False, "rocprof not available", []
        
        cmd = self.build_command(workload_cmd)
        logger.info(f"Running rocprof: {' '.join(cmd[:10])}...")
        
        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        
        # Ensure GPU is visible
        if "HIP_VISIBLE_DEVICES" not in run_env:
            run_env["HIP_VISIBLE_DEVICES"] = "0"
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env,
                cwd=str(self.config.output_dir),
            )
            
            output = result.stdout + "\n" + result.stderr
            success = result.returncode == 0
            
            if not success:
                logger.error(f"rocprof failed: {result.stderr[:500]}")
            
            # Find output files
            self._output_files = self._find_output_files()
            
            return success, output, self._output_files
            
        except subprocess.TimeoutExpired:
            logger.error(f"rocprof timed out after {timeout}s")
            return False, "Timeout", []
        except Exception as e:
            logger.error(f"rocprof error: {e}")
            return False, str(e), []
    
    def _find_output_files(self) -> List[Path]:
        """Find all rocprof output files in the output directory."""
        files = []
        
        for pattern in ["*.csv", "*.json", "*.txt", "*.stats.csv", "*results.csv"]:
            files.extend(self.config.output_dir.glob(pattern))
        
        # Also look in rocprof subdirectory
        rocprof_dir = self.config.output_dir / "rocprof"
        if rocprof_dir.exists():
            for pattern in ["*.csv", "*.json", "*.txt"]:
                files.extend(rocprof_dir.glob(pattern))
        
        return list(set(files))
    
    def get_kernel_csv(self) -> Optional[Path]:
        """Get the path to the kernel timing CSV file."""
        candidates = [
            self.config.output_dir / "rocprof_results.csv",
            self.config.output_dir / "rocprof.csv",
            self.config.output_dir / "results.csv",
        ]
        
        for f in candidates:
            if f.exists():
                return f
        
        # Search for any CSV with 'results' or 'kernel' in name
        for f in self._output_files:
            if "results" in f.name.lower() or "kernel" in f.name.lower():
                return f
        
        # Return first CSV found
        csvs = [f for f in self._output_files if f.suffix == ".csv"]
        return csvs[0] if csvs else None
    
    def get_stats_csv(self) -> Optional[Path]:
        """Get the path to the stats summary CSV file."""
        candidates = [
            self.config.output_dir / "rocprof_results.stats.csv",
            self.config.output_dir / "rocprof.stats.csv",
        ]
        
        for f in candidates:
            if f.exists():
                return f
        
        # Search for stats file
        for f in self._output_files:
            if ".stats" in f.name.lower():
                return f
        
        return None
    
    def cleanup(self) -> None:
        """Remove temporary profiling files."""
        for f in self._output_files:
            try:
                f.unlink()
            except OSError:
                pass


def profile_command(
    cmd: List[str],
    output_dir: str,
    **kwargs,
) -> Tuple[bool, List[Path]]:
    """
    Convenience function to profile a command.
    
    Args:
        cmd: Command to profile
        output_dir: Directory for output files
        **kwargs: Additional RocprofConfig options
        
    Returns:
        Tuple of (success, output_files)
    """
    config = RocprofConfig(output_dir=Path(output_dir), **kwargs)
    wrapper = RocprofWrapper(config)
    
    success, _, files = wrapper.run(cmd)
    return success, files
