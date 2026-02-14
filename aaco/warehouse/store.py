"""
Fleet-scale Performance Warehouse.

SQLite-based storage for benchmark results with:
- Multi-dimensional indexing (model, backend, GPU, ROCm version)
- Trend analytics
- Session management
- Fleet aggregation queries
"""

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class SessionMetadata:
    """Metadata for a benchmark session."""

    session_id: str
    timestamp_utc: float

    # Model info
    model_name: str = ""
    model_format: str = ""  # onnx, gguf, etc.

    # Backend info
    backend: str = ""  # pytorch, onnxruntime, llama.cpp
    backend_version: str = ""

    # Hardware info
    gpu_name: str = ""
    gpu_id: int = 0
    gpu_memory_gb: float = 0.0

    # Software environment
    rocm_version: str = ""
    driver_version: str = ""
    os_info: str = ""

    # Run configuration
    batch_size: int = 1
    precision: str = "fp16"
    optimization_level: str = ""

    # Tags for filtering
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """A single benchmark result."""

    result_id: str
    session_id: str
    timestamp_utc: float

    # Metric
    metric_name: str  # e.g., "latency_p50_ms", "throughput_tok_s"
    metric_value: float
    metric_unit: str = ""

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendPoint:
    """A point in a trend series."""

    timestamp: float
    value: float
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Fleet Warehouse
# ============================================================================


class FleetWarehouse:
    """
    Fleet-scale performance warehouse.

    Stores benchmark results in SQLite with efficient indexing
    for trend analytics and cross-fleet comparisons.

    Schema:
    - sessions: Benchmark session metadata
    - results: Individual metric results
    - kernel_stats: Aggregated kernel statistics

    Usage:
        warehouse = FleetWarehouse(Path("perf_warehouse.db"))

        # Start a session
        session = warehouse.create_session(
            model_name="llama-7b",
            backend="pytorch",
            gpu_name="RX 7900 XTX",
        )

        # Record results
        warehouse.record_metric(session.session_id, "latency_p50_ms", 45.2)
        warehouse.record_metric(session.session_id, "throughput_tok_s", 125.0)

        # Query trends
        trend = warehouse.get_trend("latency_p50_ms", model_name="llama-7b")
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp_utc REAL NOT NULL,
                    model_name TEXT,
                    model_format TEXT,
                    backend TEXT,
                    backend_version TEXT,
                    gpu_name TEXT,
                    gpu_id INTEGER,
                    gpu_memory_gb REAL,
                    rocm_version TEXT,
                    driver_version TEXT,
                    os_info TEXT,
                    batch_size INTEGER,
                    precision TEXT,
                    optimization_level TEXT,
                    tags TEXT,
                    metadata_json TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_sessions_model 
                    ON sessions(model_name);
                CREATE INDEX IF NOT EXISTS idx_sessions_backend 
                    ON sessions(backend);
                CREATE INDEX IF NOT EXISTS idx_sessions_gpu 
                    ON sessions(gpu_name);
                CREATE INDEX IF NOT EXISTS idx_sessions_timestamp 
                    ON sessions(timestamp_utc);
                
                CREATE TABLE IF NOT EXISTS results (
                    result_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp_utc REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    context_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_results_session 
                    ON results(session_id);
                CREATE INDEX IF NOT EXISTS idx_results_metric 
                    ON results(metric_name);
                CREATE INDEX IF NOT EXISTS idx_results_timestamp 
                    ON results(timestamp_utc);
                
                CREATE TABLE IF NOT EXISTS kernel_stats (
                    stat_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    kernel_name TEXT NOT NULL,
                    call_count INTEGER,
                    total_time_us REAL,
                    mean_time_us REAL,
                    min_time_us REAL,
                    max_time_us REAL,
                    std_time_us REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_kernel_stats_session 
                    ON kernel_stats(session_id);
                CREATE INDEX IF NOT EXISTS idx_kernel_stats_kernel 
                    ON kernel_stats(kernel_name);
            """)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ==========================================================================
    # Session Management
    # ==========================================================================

    def create_session(self, **kwargs) -> SessionMetadata:
        """
        Create a new benchmark session.

        Args:
            **kwargs: Session metadata fields

        Returns:
            Session metadata with generated ID
        """
        session_id = self._generate_session_id()
        timestamp = time.time()

        metadata = SessionMetadata(session_id=session_id, timestamp_utc=timestamp, **kwargs)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, timestamp_utc, model_name, model_format,
                    backend, backend_version, gpu_name, gpu_id, gpu_memory_gb,
                    rocm_version, driver_version, os_info, batch_size,
                    precision, optimization_level, tags, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata.session_id,
                    metadata.timestamp_utc,
                    metadata.model_name,
                    metadata.model_format,
                    metadata.backend,
                    metadata.backend_version,
                    metadata.gpu_name,
                    metadata.gpu_id,
                    metadata.gpu_memory_gb,
                    metadata.rocm_version,
                    metadata.driver_version,
                    metadata.os_info,
                    metadata.batch_size,
                    metadata.precision,
                    metadata.optimization_level,
                    json.dumps(metadata.tags),
                    json.dumps(metadata.to_dict()),
                ),
            )

        logger.info(f"Created session: {session_id}")
        return metadata

    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """Get session metadata by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

        if row:
            data = json.loads(row["metadata_json"])
            return SessionMetadata(**data)
        return None

    def list_sessions(
        self,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        gpu_name: Optional[str] = None,
        since_days: Optional[int] = None,
        limit: int = 100,
    ) -> List[SessionMetadata]:
        """
        List sessions matching filters.

        Args:
            model_name: Filter by model name
            backend: Filter by backend
            gpu_name: Filter by GPU
            since_days: Only sessions from last N days
            limit: Maximum results

        Returns:
            List of matching sessions
        """
        query = "SELECT metadata_json FROM sessions WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if backend:
            query += " AND backend = ?"
            params.append(backend)

        if gpu_name:
            query += " AND gpu_name = ?"
            params.append(gpu_name)

        if since_days:
            cutoff = time.time() - (since_days * 86400)
            query += " AND timestamp_utc > ?"
            params.append(cutoff)

        query += " ORDER BY timestamp_utc DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [SessionMetadata(**json.loads(row["metadata_json"])) for row in rows]

    # ==========================================================================
    # Metric Recording
    # ==========================================================================

    def record_metric(
        self,
        session_id: str,
        metric_name: str,
        metric_value: float,
        metric_unit: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a benchmark metric.

        Args:
            session_id: Session ID
            metric_name: Metric name (e.g., "latency_p50_ms")
            metric_value: Metric value
            metric_unit: Unit (optional)
            context: Additional context

        Returns:
            Result ID
        """
        result_id = self._generate_result_id(session_id, metric_name)
        timestamp = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO results (
                    result_id, session_id, timestamp_utc,
                    metric_name, metric_value, metric_unit, context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result_id,
                    session_id,
                    timestamp,
                    metric_name,
                    metric_value,
                    metric_unit,
                    json.dumps(context or {}),
                ),
            )

        return result_id

    def record_metrics(self, session_id: str, metrics: Dict[str, float]) -> None:
        """Record multiple metrics at once."""
        for name, value in metrics.items():
            self.record_metric(session_id, name, value)

    def record_kernel_stats(
        self,
        session_id: str,
        kernel_name: str,
        call_count: int,
        total_time_us: float,
        mean_time_us: float,
        min_time_us: float,
        max_time_us: float,
        std_time_us: float = 0.0,
    ) -> None:
        """Record aggregated kernel statistics."""
        stat_id = f"{session_id}_{hashlib.md5(kernel_name.encode()).hexdigest()[:8]}"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO kernel_stats (
                    stat_id, session_id, kernel_name,
                    call_count, total_time_us, mean_time_us,
                    min_time_us, max_time_us, std_time_us
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    stat_id,
                    session_id,
                    kernel_name,
                    call_count,
                    total_time_us,
                    mean_time_us,
                    min_time_us,
                    max_time_us,
                    std_time_us,
                ),
            )

    # ==========================================================================
    # Trend Analytics
    # ==========================================================================

    def get_trend(
        self,
        metric_name: str,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        gpu_name: Optional[str] = None,
        since_days: int = 30,
        limit: int = 1000,
    ) -> List[TrendPoint]:
        """
        Get trend data for a metric.

        Args:
            metric_name: Metric to query
            model_name: Filter by model
            backend: Filter by backend
            gpu_name: Filter by GPU
            since_days: Days of history
            limit: Max data points

        Returns:
            List of trend points ordered by time
        """
        cutoff = time.time() - (since_days * 86400)

        query = """
            SELECT r.timestamp_utc, r.metric_value, r.session_id,
                   s.model_name, s.backend, s.gpu_name
            FROM results r
            JOIN sessions s ON r.session_id = s.session_id
            WHERE r.metric_name = ? AND r.timestamp_utc > ?
        """
        params: List[Any] = [metric_name, cutoff]

        if model_name:
            query += " AND s.model_name = ?"
            params.append(model_name)

        if backend:
            query += " AND s.backend = ?"
            params.append(backend)

        if gpu_name:
            query += " AND s.gpu_name = ?"
            params.append(gpu_name)

        query += " ORDER BY r.timestamp_utc ASC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            TrendPoint(
                timestamp=row["timestamp_utc"],
                value=row["metric_value"],
                session_id=row["session_id"],
                metadata={
                    "model_name": row["model_name"],
                    "backend": row["backend"],
                    "gpu_name": row["gpu_name"],
                },
            )
            for row in rows
        ]

    def get_metric_stats(
        self,
        metric_name: str,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        since_days: int = 7,
    ) -> Dict[str, float]:
        """
        Get aggregate statistics for a metric.

        Returns:
            Dictionary with mean, std, min, max, count
        """
        cutoff = time.time() - (since_days * 86400)

        query = """
            SELECT 
                AVG(r.metric_value) as mean,
                MIN(r.metric_value) as min,
                MAX(r.metric_value) as max,
                COUNT(*) as count
            FROM results r
            JOIN sessions s ON r.session_id = s.session_id
            WHERE r.metric_name = ? AND r.timestamp_utc > ?
        """
        params: List[Any] = [metric_name, cutoff]

        if model_name:
            query += " AND s.model_name = ?"
            params.append(model_name)

        if backend:
            query += " AND s.backend = ?"
            params.append(backend)

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()

        if row and row["count"] > 0:
            return {
                "mean": row["mean"],
                "min": row["min"],
                "max": row["max"],
                "count": row["count"],
            }
        return {"mean": 0, "min": 0, "max": 0, "count": 0}

    def compare_backends(
        self, metric_name: str, model_name: str, since_days: int = 7
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metric across backends for a model.

        Returns:
            Dictionary mapping backend -> stats
        """
        cutoff = time.time() - (since_days * 86400)

        query = """
            SELECT 
                s.backend,
                AVG(r.metric_value) as mean,
                MIN(r.metric_value) as min,
                MAX(r.metric_value) as max,
                COUNT(*) as count
            FROM results r
            JOIN sessions s ON r.session_id = s.session_id
            WHERE r.metric_name = ? 
              AND s.model_name = ?
              AND r.timestamp_utc > ?
            GROUP BY s.backend
        """

        with self._get_connection() as conn:
            rows = conn.execute(query, (metric_name, model_name, cutoff)).fetchall()

        return {
            row["backend"]: {
                "mean": row["mean"],
                "min": row["min"],
                "max": row["max"],
                "count": row["count"],
            }
            for row in rows
        }

    def compare_gpus(
        self, metric_name: str, model_name: str, backend: str, since_days: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metric across GPUs for a model/backend.

        Returns:
            Dictionary mapping gpu_name -> stats
        """
        cutoff = time.time() - (since_days * 86400)

        query = """
            SELECT 
                s.gpu_name,
                AVG(r.metric_value) as mean,
                MIN(r.metric_value) as min,
                MAX(r.metric_value) as max,
                COUNT(*) as count
            FROM results r
            JOIN sessions s ON r.session_id = s.session_id
            WHERE r.metric_name = ?
              AND s.model_name = ?
              AND s.backend = ?
              AND r.timestamp_utc > ?
            GROUP BY s.gpu_name
        """

        with self._get_connection() as conn:
            rows = conn.execute(query, (metric_name, model_name, backend, cutoff)).fetchall()

        return {
            row["gpu_name"]: {
                "mean": row["mean"],
                "min": row["min"],
                "max": row["max"],
                "count": row["count"],
            }
            for row in rows
        }

    # ==========================================================================
    # Regression Detection
    # ==========================================================================

    def detect_regression(
        self,
        metric_name: str,
        model_name: str,
        baseline_days: int = 7,
        current_days: int = 1,
        threshold_pct: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect regression by comparing recent results to baseline.

        Args:
            metric_name: Metric to check
            model_name: Model name
            baseline_days: Days for baseline period
            current_days: Days for current period
            threshold_pct: Regression threshold percentage

        Returns:
            Regression info if detected, None otherwise
        """
        now = time.time()

        # Get baseline stats
        baseline_cutoff = now - (baseline_days * 86400)
        current_cutoff = now - (current_days * 86400)

        query = """
            SELECT 
                AVG(r.metric_value) as mean,
                COUNT(*) as count
            FROM results r
            JOIN sessions s ON r.session_id = s.session_id
            WHERE r.metric_name = ?
              AND s.model_name = ?
              AND r.timestamp_utc BETWEEN ? AND ?
        """

        with self._get_connection() as conn:
            baseline_row = conn.execute(
                query, (metric_name, model_name, baseline_cutoff, current_cutoff)
            ).fetchone()

            current_row = conn.execute(
                query, (metric_name, model_name, current_cutoff, now)
            ).fetchone()

        if not baseline_row or not current_row:
            return None

        if baseline_row["count"] < 3 or current_row["count"] < 1:
            return None

        baseline_mean = baseline_row["mean"]
        current_mean = current_row["mean"]

        if baseline_mean == 0:
            return None

        delta_pct = ((current_mean - baseline_mean) / baseline_mean) * 100

        # For latency metrics, increase = regression
        # For throughput metrics, decrease = regression
        is_latency = "latency" in metric_name.lower() or "time" in metric_name.lower()
        is_regression = delta_pct > threshold_pct if is_latency else delta_pct < -threshold_pct

        if is_regression:
            return {
                "metric_name": metric_name,
                "model_name": model_name,
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "delta_pct": delta_pct,
                "threshold_pct": threshold_pct,
            }

        return None

    # ==========================================================================
    # Utilities
    # ==========================================================================

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        ts = int(time.time() * 1000)
        rand = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"sess_{ts}_{rand}"

    def _generate_result_id(self, session_id: str, metric_name: str) -> str:
        """Generate unique result ID."""
        ts = int(time.time() * 1000000)  # Microseconds
        return f"res_{ts}_{hashlib.md5(f'{session_id}_{metric_name}'.encode()).hexdigest()[:6]}"

    def get_stats(self) -> Dict[str, int]:
        """Get warehouse statistics."""
        with self._get_connection() as conn:
            session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            result_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            kernel_count = conn.execute("SELECT COUNT(*) FROM kernel_stats").fetchone()[0]

        return {
            "sessions": session_count,
            "results": result_count,
            "kernel_stats": kernel_count,
        }


# ============================================================================
# Convenience Functions
# ============================================================================


def create_warehouse(path: Path = Path("aaco_warehouse.db")) -> FleetWarehouse:
    """Create a new warehouse."""
    return FleetWarehouse(path)


def get_default_warehouse() -> FleetWarehouse:
    """Get the default warehouse."""
    default_path = Path.home() / ".aaco" / "warehouse.db"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return FleetWarehouse(default_path)
