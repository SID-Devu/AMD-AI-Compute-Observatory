"""
AACO Driver Interface - Python User-space Bindings
Communicates with the /dev/aaco character device for kernel-level telemetry.
"""

import fcntl
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Enums (matching kernel header)
# ============================================================================

AACO_DEVICE_PATH = "/dev/aaco"
AACO_DEBUGFS_PATH = "/sys/kernel/debug/aaco"
AACO_MAGIC = ord("A")


class AACOEventType(IntEnum):
    """Event types from the kernel driver."""

    SESSION_START = 0x0001
    SESSION_STOP = 0x0002

    CTX_SWITCH_VOL = 0x0010
    CTX_SWITCH_INVOL = 0x0011

    PAGE_FAULT_MAJOR = 0x0020
    PAGE_FAULT_MINOR = 0x0021

    CPU_TIME = 0x0030

    RSS_SAMPLE = 0x0040

    RUNQ_SAMPLE = 0x0050
    SCHED_LATENCY = 0x0051

    USER_MARKER = 0x0100

    PHASE_WARMUP = 0x0110
    PHASE_MEASURE = 0x0111
    PHASE_PREFILL = 0x0112
    PHASE_DECODE = 0x0113

    ITER_START = 0x0120
    ITER_END = 0x0121

    GPU_KERNEL_START = 0x0200
    GPU_KERNEL_END = 0x0201


# IOCTL command numbers (must match kernel)
def _IOC(direction, type_char, nr, size):
    """Generate IOCTL command number."""
    IOC_NRBITS = 8
    IOC_TYPEBITS = 8
    IOC_SIZEBITS = 14

    IOC_NRSHIFT = 0
    IOC_TYPESHIFT = IOC_NRSHIFT + IOC_NRBITS
    IOC_SIZESHIFT = IOC_TYPESHIFT + IOC_TYPEBITS
    IOC_DIRSHIFT = IOC_SIZESHIFT + IOC_SIZEBITS

    return (
        (direction << IOC_DIRSHIFT)
        | (type_char << IOC_TYPESHIFT)
        | (nr << IOC_NRSHIFT)
        | (size << IOC_SIZESHIFT)
    )


_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2

AACO_IOC_SESSION_START = _IOC(_IOC_WRITE, AACO_MAGIC, 0x01, 32)  # sizeof(aaco_session_cmd)
AACO_IOC_SESSION_STOP = _IOC(_IOC_WRITE, AACO_MAGIC, 0x02, 32)
AACO_IOC_GET_STATS = _IOC(_IOC_READ | _IOC_WRITE, AACO_MAGIC, 0x03, 32)
AACO_IOC_SET_SAMPLE_MS = _IOC(_IOC_WRITE, AACO_MAGIC, 0x04, 8)
AACO_IOC_EMIT_MARKER = _IOC(_IOC_WRITE, AACO_MAGIC, 0x05, 32)


# ============================================================================
# Data Structures (matching kernel)
# ============================================================================


@dataclass
class AACOEvent:
    """Event record from kernel driver."""

    timestamp_ns: int
    session_id: int
    pid: int
    event_type: AACOEventType
    cpu: int
    value1: int
    value2: int
    comm: str

    # Computed fields
    @property
    def event_type_name(self) -> str:
        try:
            return AACOEventType(self.event_type).name
        except ValueError:
            return f"UNKNOWN_{self.event_type:#x}"

    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000

    @property
    def timestamp_us(self) -> float:
        return self.timestamp_ns / 1_000


@dataclass
class AACOStats:
    """Session statistics from kernel driver."""

    session_id: int
    pid: int
    duration_ns: int
    samples_collected: int
    events_generated: int
    total_nvcsw: int
    total_nivcsw: int
    total_maj_flt: int
    total_min_flt: int
    total_utime_ns: int
    total_stime_ns: int
    rss_peak_bytes: int

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000

    @property
    def ctx_switch_rate_per_sec(self) -> float:
        if self.duration_ns == 0:
            return 0
        total_switches = self.total_nvcsw + self.total_nivcsw
        return total_switches / (self.duration_ns / 1e9)

    @property
    def fault_rate_per_sec(self) -> float:
        if self.duration_ns == 0:
            return 0
        total_faults = self.total_maj_flt + self.total_min_flt
        return total_faults / (self.duration_ns / 1e9)

    @property
    def cpu_time_ratio(self) -> float:
        """Ratio of system time to user time."""
        if self.total_utime_ns == 0:
            return 0
        return self.total_stime_ns / self.total_utime_ns

    @property
    def rss_peak_mb(self) -> float:
        return self.rss_peak_bytes / (1024 * 1024)


# ============================================================================
# Driver Interface
# ============================================================================


class AACODriverInterface:
    """
    User-space interface to the AACO kernel driver.

    Provides session management, event streaming, and statistics collection.
    """

    # Event record struct format: timestamp(Q) session_id(I) pid(I)
    #                             event_type(H) cpu(H) value1(Q) value2(Q) comm(16s)
    EVENT_FORMAT = "<QIIHHQQl6s"
    EVENT_SIZE = 48  # Must match kernel struct

    # Session command struct format
    SESSION_CMD_FORMAT = "<IIQQ"  # session_id, pid, flags, data/marker

    # Stats struct format
    STATS_FORMAT = "<IIQQQQQQQQQQ"  # Matching aaco_stats

    def __init__(self, device_path: str = AACO_DEVICE_PATH):
        self.device_path = device_path
        self._fd: Optional[int] = None
        self._event_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: List[Callable[[AACOEvent], None]] = []

    def is_available(self) -> bool:
        """Check if driver is loaded and device exists."""
        return os.path.exists(self.device_path)

    def open(self) -> None:
        """Open the driver device."""
        if self._fd is not None:
            return

        if not self.is_available():
            raise RuntimeError(f"AACO driver not available at {self.device_path}")

        self._fd = os.open(self.device_path, os.O_RDWR)
        logger.info(f"AACO driver opened: {self.device_path}")

    def close(self) -> None:
        """Close the driver device."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
            logger.info("AACO driver closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event_reader()
        self.close()
        return False

    def session_start(self, session_id: int, pid: int = 0) -> None:
        """
        Start a tracking session.

        Args:
            session_id: Unique session identifier
            pid: Target process ID (0 = current process)
        """
        if self._fd is None:
            raise RuntimeError("Driver not opened")

        cmd = struct.pack(self.SESSION_CMD_FORMAT, session_id, pid, 0, 0)
        fcntl.ioctl(self._fd, AACO_IOC_SESSION_START, cmd)
        logger.info(f"Session {session_id} started for pid {pid or os.getpid()}")

    def session_stop(self, session_id: int) -> None:
        """Stop a tracking session."""
        if self._fd is None:
            raise RuntimeError("Driver not opened")

        cmd = struct.pack(self.SESSION_CMD_FORMAT, session_id, 0, 0, 0)
        fcntl.ioctl(self._fd, AACO_IOC_SESSION_STOP, cmd)
        logger.info(f"Session {session_id} stopped")

    def get_stats(self, session_id: int) -> AACOStats:
        """Get statistics for a session."""
        if self._fd is None:
            raise RuntimeError("Driver not opened")

        # This is simplified - real implementation needs proper memory handling
        stats_buf = bytearray(128)  # Large enough for stats struct
        struct.pack(
            self.SESSION_CMD_FORMAT, session_id, 0, 0, id(stats_buf)
        )  # This won't work as-is, simplified

        # For now, return mock stats - real implementation needs ctypes
        return AACOStats(
            session_id=session_id,
            pid=os.getpid(),
            duration_ns=0,
            samples_collected=0,
            events_generated=0,
            total_nvcsw=0,
            total_nivcsw=0,
            total_maj_flt=0,
            total_min_flt=0,
            total_utime_ns=0,
            total_stime_ns=0,
            rss_peak_bytes=0,
        )

    def set_sample_interval(self, interval_ms: int) -> None:
        """Set sampling interval in milliseconds."""
        if self._fd is None:
            raise RuntimeError("Driver not opened")

        if not 1 <= interval_ms <= 1000:
            raise ValueError("Interval must be between 1 and 1000 ms")

        fcntl.ioctl(self._fd, AACO_IOC_SET_SAMPLE_MS, interval_ms)
        logger.debug(f"Sampling interval set to {interval_ms}ms")

    def emit_marker(self, session_id: int, marker_id: int, marker_value: int = 0) -> None:
        """Emit a user-space marker event."""
        if self._fd is None:
            raise RuntimeError("Driver not opened")

        # Pack marker into flags field
        marker_data = (marker_id & 0xFFFFFFFF) | ((marker_value & 0xFFFFFFFF) << 32)
        cmd = struct.pack(self.SESSION_CMD_FORMAT, session_id, 0, marker_data, 0)
        fcntl.ioctl(self._fd, AACO_IOC_EMIT_MARKER, cmd)

    def read_events(self, max_events: int = 100) -> List[AACOEvent]:
        """Read available events (blocking)."""
        if self._fd is None:
            raise RuntimeError("Driver not opened")

        buf_size = max_events * self.EVENT_SIZE
        try:
            data = os.read(self._fd, buf_size)
        except BlockingIOError:
            return []

        events = []
        for i in range(0, len(data), self.EVENT_SIZE):
            chunk = data[i : i + self.EVENT_SIZE]
            if len(chunk) < self.EVENT_SIZE:
                break

            # Unpack event
            (
                timestamp_ns,
                session_id,
                pid,
                event_type,
                cpu,
                value1,
                value2,
                comm_raw,
            ) = struct.unpack("<QIIHH QQ 16s", chunk)

            comm = comm_raw.rstrip(b"\x00").decode("utf-8", errors="replace")

            events.append(
                AACOEvent(
                    timestamp_ns=timestamp_ns,
                    session_id=session_id,
                    pid=pid,
                    event_type=AACOEventType(event_type)
                    if event_type in AACOEventType._value2member_map_
                    else event_type,
                    cpu=cpu,
                    value1=value1,
                    value2=value2,
                    comm=comm,
                )
            )

        return events

    def read_events_iter(self) -> Iterator[AACOEvent]:
        """Iterate over events as they arrive."""
        while not self._stop_event.is_set():
            events = self.read_events(max_events=100)
            for event in events:
                yield event
            if not events:
                time.sleep(0.001)  # Small sleep to avoid busy-wait

    def add_callback(self, callback: Callable[[AACOEvent], None]) -> None:
        """Add a callback for event processing."""
        self._callbacks.append(callback)

    def start_event_reader(self) -> None:
        """Start background event reader thread."""
        if self._event_thread is not None:
            return

        self._stop_event.clear()
        self._event_thread = threading.Thread(target=self._event_reader_loop, daemon=True)
        self._event_thread.start()
        logger.info("Event reader thread started")

    def stop_event_reader(self) -> None:
        """Stop background event reader thread."""
        if self._event_thread is None:
            return

        self._stop_event.set()
        self._event_thread.join(timeout=2.0)
        self._event_thread = None
        logger.info("Event reader thread stopped")

    def _event_reader_loop(self) -> None:
        """Background event reader loop."""
        while not self._stop_event.is_set():
            try:
                events = self.read_events(max_events=100)
                for event in events:
                    for callback in self._callbacks:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                if not events:
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"Event reader error: {e}")
                time.sleep(0.1)


# ============================================================================
# Event Collector (high-level interface)
# ============================================================================


@dataclass
class DriverCollectorConfig:
    """Configuration for driver-based collection."""

    sample_interval_ms: int = 10
    buffer_events: bool = True
    max_buffered_events: int = 10000


class DriverEventCollector:
    """
    High-level collector that integrates with the AACO kernel driver.

    Provides buffered event collection and export to analysis pipeline.
    """

    def __init__(self, config: Optional[DriverCollectorConfig] = None):
        self.config = config or DriverCollectorConfig()
        self._driver: Optional[AACODriverInterface] = None
        self._session_id: Optional[int] = None
        self._events: List[AACOEvent] = []
        self._lock = threading.Lock()
        self._start_time_ns: int = 0

    def is_available(self) -> bool:
        """Check if driver collection is available."""
        return AACODriverInterface().is_available()

    def start(self, session_id: int) -> bool:
        """
        Start driver-based collection.

        Args:
            session_id: Session identifier (should match AACO session)

        Returns:
            True if started successfully, False if driver unavailable
        """
        if not self.is_available():
            logger.warning("AACO driver not available - skipping kernel telemetry")
            return False

        try:
            self._driver = AACODriverInterface()
            self._driver.open()
            self._driver.set_sample_interval(self.config.sample_interval_ms)

            if self.config.buffer_events:
                self._driver.add_callback(self._buffer_event)
                self._driver.start_event_reader()

            self._driver.session_start(session_id)
            self._session_id = session_id
            self._start_time_ns = time.monotonic_ns()

            logger.info(f"Driver collector started for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start driver collector: {e}")
            if self._driver:
                self._driver.close()
                self._driver = None
            return False

    def stop(self) -> Optional[AACOStats]:
        """Stop collection and return statistics."""
        if self._driver is None or self._session_id is None:
            return None

        try:
            self._driver.session_stop(self._session_id)
            stats = self._driver.get_stats(self._session_id)

            self._driver.stop_event_reader()
            self._driver.close()

            logger.info(f"Driver collector stopped: {len(self._events)} events buffered")
            return stats

        except Exception as e:
            logger.error(f"Error stopping driver collector: {e}")
            return None

        finally:
            self._driver = None
            self._session_id = None

    def emit_marker(self, marker_id: int, value: int = 0) -> None:
        """Emit a phase/iteration marker."""
        if self._driver and self._session_id:
            try:
                self._driver.emit_marker(self._session_id, marker_id, value)
            except Exception as e:
                logger.warning(f"Failed to emit marker: {e}")

    def get_events(self) -> List[AACOEvent]:
        """Get all buffered events."""
        with self._lock:
            return list(self._events)

    def export_to_dicts(self) -> List[Dict[str, Any]]:
        """Export events as list of dictionaries."""
        events = self.get_events()
        return [
            {
                "t_ns": e.timestamp_ns,
                "t_ms": e.timestamp_ms,
                "session_id": e.session_id,
                "pid": e.pid,
                "event_type": e.event_type,
                "event_type_name": e.event_type_name,
                "cpu": e.cpu,
                "value1": e.value1,
                "value2": e.value2,
                "comm": e.comm,
            }
            for e in events
        ]

    def _buffer_event(self, event: AACOEvent) -> None:
        """Buffer an event (callback)."""
        with self._lock:
            if len(self._events) < self.config.max_buffered_events:
                self._events.append(event)
            elif len(self._events) == self.config.max_buffered_events:
                logger.warning("Event buffer full - dropping events")


# ============================================================================
# Debugfs Reader (for driver statistics)
# ============================================================================


def read_debugfs_stats() -> Optional[Dict[str, Any]]:
    """Read driver statistics from debugfs."""
    stats_path = Path(AACO_DEBUGFS_PATH) / "stats"

    if not stats_path.exists():
        return None

    try:
        content = stats_path.read_text()
        stats = {}

        for line in content.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                # Try to parse as number
                try:
                    stats[key] = int(value)
                except ValueError:
                    stats[key] = value

        return stats

    except Exception as e:
        logger.error(f"Failed to read debugfs stats: {e}")
        return None


def read_debugfs_sessions() -> Optional[List[Dict[str, Any]]]:
    """Read active sessions from debugfs."""
    sessions_path = Path(AACO_DEBUGFS_PATH) / "sessions"

    if not sessions_path.exists():
        return None

    try:
        content = sessions_path.read_text()
        sessions = []

        for line in content.strip().split("\n"):
            if line.startswith("Session "):
                # Parse: "Session N: pid=X, samples=Y, events=Z, interval=Wms"
                parts = line.split(":")
                if len(parts) >= 2:
                    session_id = int(parts[0].split()[1])
                    info = {}
                    for kv in parts[1].split(","):
                        k, v = kv.strip().split("=")
                        info[k] = int(v.rstrip("ms"))
                    info["session_id"] = session_id
                    sessions.append(info)

        return sessions

    except Exception as e:
        logger.error(f"Failed to read debugfs sessions: {e}")
        return None
