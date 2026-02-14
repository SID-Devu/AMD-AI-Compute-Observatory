"""
AACO Utilities
Common utilities for timing, subprocess management, file I/O, and system inspection.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_monotonic_ns() -> int:
    """Get monotonic clock time in nanoseconds."""
    return time.monotonic_ns()


def get_monotonic_ms() -> float:
    """Get monotonic clock time in milliseconds."""
    return time.monotonic_ns() / 1_000_000


def ns_to_ms(ns: int) -> float:
    """Convert nanoseconds to milliseconds."""
    return ns / 1_000_000


def ns_to_us(ns: int) -> float:
    """Convert nanoseconds to microseconds."""
    return ns / 1_000


def ms_to_ns(ms: float) -> int:
    """Convert milliseconds to nanoseconds."""
    return int(ms * 1_000_000)


def run_command(
    cmd: List[str],
    timeout: int = 30,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    capture_stderr: bool = True,
) -> Optional[str]:
    """
    Run a shell command and return stdout.
    Returns None on failure.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env or os.environ.copy(),
        )
        if result.returncode == 0:
            return result.stdout
        else:
            if capture_stderr:
                logger.debug(f"Command failed: {cmd}\nstderr: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {cmd}")
        return None
    except FileNotFoundError:
        logger.debug(f"Command not found: {cmd[0]}")
        return None
    except Exception as e:
        logger.debug(f"Command error: {cmd}, {e}")
        return None


def run_command_async(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    """Start a command asynchronously and return the Popen object."""
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env or os.environ.copy(),
    )


def safe_json_dump(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Safely write JSON to file with atomic write pattern."""
    path = Path(path)
    temp_path = path.with_suffix(".tmp")

    try:
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=indent, default=str)
        temp_path.replace(path)
    except Exception as e:
        logger.error(f"Failed to write JSON to {path}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise


def safe_json_load(path: Union[str, Path]) -> Optional[Any]:
    """Safely load JSON from file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        return None


def read_proc_file(path: str) -> Optional[str]:
    """Read a /proc or /sys file, return None on failure."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return None


def read_proc_stat() -> Dict[str, Any]:
    """Read /proc/stat and parse CPU statistics."""
    content = read_proc_file("/proc/stat")
    if not content:
        return {}

    stats = {}
    for line in content.split("\n"):
        if line.startswith("cpu "):
            parts = line.split()
            stats["cpu_total"] = {
                "user": int(parts[1]),
                "nice": int(parts[2]),
                "system": int(parts[3]),
                "idle": int(parts[4]),
                "iowait": int(parts[5]) if len(parts) > 5 else 0,
                "irq": int(parts[6]) if len(parts) > 6 else 0,
                "softirq": int(parts[7]) if len(parts) > 7 else 0,
            }
        elif line.startswith("ctxt"):
            stats["context_switches"] = int(line.split()[1])
        elif line.startswith("procs_running"):
            stats["procs_running"] = int(line.split()[1])
        elif line.startswith("procs_blocked"):
            stats["procs_blocked"] = int(line.split()[1])

    return stats


def read_proc_vmstat() -> Dict[str, int]:
    """Read /proc/vmstat and parse memory statistics."""
    content = read_proc_file("/proc/vmstat")
    if not content:
        return {}

    stats = {}
    for line in content.split("\n"):
        parts = line.split()
        if len(parts) == 2:
            try:
                stats[parts[0]] = int(parts[1])
            except ValueError:
                pass

    return stats


def read_loadavg() -> Dict[str, float]:
    """Read /proc/loadavg."""
    content = read_proc_file("/proc/loadavg")
    if not content:
        return {}

    parts = content.split()
    return {
        "load1": float(parts[0]),
        "load5": float(parts[1]),
        "load15": float(parts[2]),
    }


def read_process_stat(pid: int) -> Dict[str, Any]:
    """Read /proc/<pid>/stat for process-specific statistics."""
    content = read_proc_file(f"/proc/{pid}/stat")
    if not content:
        return {}

    # Parse the complex /proc/pid/stat format
    # Format: pid (comm) state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt utime stime ...
    parts = content.split()

    return {
        "pid": int(parts[0]),
        "state": parts[2],
        "minflt": int(parts[9]),
        "majflt": int(parts[11]),
        "utime": int(parts[13]),
        "stime": int(parts[14]),
        "num_threads": int(parts[19]) if len(parts) > 19 else 1,
        "vsize": int(parts[22]) if len(parts) > 22 else 0,
        "rss": int(parts[23]) if len(parts) > 23 else 0,
    }


def read_process_status(pid: int) -> Dict[str, Any]:
    """Read /proc/<pid>/status for detailed process info."""
    content = read_proc_file(f"/proc/{pid}/status")
    if not content:
        return {}

    status = {}
    for line in content.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Parse specific fields
            if key in ["VmRSS", "VmSize", "VmPeak"]:
                # Remove "kB" suffix and convert
                try:
                    status[key] = int(value.split()[0])
                except (ValueError, IndexError):
                    pass
            elif key in ["voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"]:
                try:
                    status[key] = int(value)
                except ValueError:
                    pass
            elif key == "Threads":
                try:
                    status["threads"] = int(value)
                except ValueError:
                    pass

    return status


def compute_cpu_percent(prev_stat: Dict, curr_stat: Dict) -> float:
    """Compute CPU percentage between two /proc/stat snapshots."""
    if not prev_stat or not curr_stat:
        return 0.0

    prev_cpu = prev_stat.get("cpu_total", {})
    curr_cpu = curr_stat.get("cpu_total", {})

    if not prev_cpu or not curr_cpu:
        return 0.0

    prev_idle = prev_cpu.get("idle", 0) + prev_cpu.get("iowait", 0)
    curr_idle = curr_cpu.get("idle", 0) + curr_cpu.get("iowait", 0)

    prev_total = sum(prev_cpu.values())
    curr_total = sum(curr_cpu.values())

    total_diff = curr_total - prev_total
    idle_diff = curr_idle - prev_idle

    if total_diff == 0:
        return 0.0

    return 100.0 * (1.0 - idle_diff / total_diff)


def format_size(bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes) < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} h"


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of a list of values."""
    if not data:
        return 0.0

    import numpy as np

    return float(np.percentile(data, p))


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_ns = 0
        self.end_ns = 0
        self.duration_ns = 0
        self.duration_ms = 0.0

    def __enter__(self) -> "Timer":
        self.start_ns = get_monotonic_ns()
        return self

    def __exit__(self, *args) -> None:
        self.end_ns = get_monotonic_ns()
        self.duration_ns = self.end_ns - self.start_ns
        self.duration_ms = ns_to_ms(self.duration_ns)

        if self.name:
            logger.debug(f"{self.name}: {self.duration_ms:.2f} ms")


class RateTracker:
    """Track rates over a sliding time window."""

    def __init__(self, window_seconds: float = 1.0):
        self.window_ns = int(window_seconds * 1e9)
        self.events: List[int] = []

    def record(self, timestamp_ns: Optional[int] = None) -> None:
        """Record an event."""
        ts = timestamp_ns or get_monotonic_ns()
        self.events.append(ts)
        self._cleanup(ts)

    def _cleanup(self, current_ns: int) -> None:
        """Remove events outside the window."""
        cutoff = current_ns - self.window_ns
        self.events = [e for e in self.events if e > cutoff]

    def get_rate(self) -> float:
        """Get events per second."""
        if not self.events:
            return 0.0

        current = get_monotonic_ns()
        self._cleanup(current)

        if not self.events:
            return 0.0

        window_actual = (current - self.events[0]) / 1e9
        if window_actual <= 0:
            return 0.0

        return len(self.events) / window_actual


def hash_file(path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    import hashlib

    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
