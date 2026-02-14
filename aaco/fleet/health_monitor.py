"""
AACO-SIGMA Health Monitor

Monitors health of GPU nodes in the fleet.
Detects issues and generates alerts.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum, auto
import time

from .fleet_manager import FleetManager, GPUNode, NodeStatus


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = auto()     # Everything normal
    WARNING = auto()     # Minor issues
    DEGRADED = auto()    # Performance degraded
    CRITICAL = auto()    # Major issues
    UNKNOWN = auto()     # Cannot determine


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class HealthMetrics:
    """Health metrics for a node."""
    
    # Availability
    uptime_pct: float = 100.0
    last_heartbeat_delay_s: float = 0.0
    
    # Performance
    job_success_rate: float = 100.0
    avg_latency_vs_baseline: float = 1.0  # 1.0 = at baseline
    
    # Resources
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    temperature_c: float = 0.0
    power_watts: float = 0.0
    
    # Errors
    error_count: int = 0
    timeout_count: int = 0


@dataclass
class NodeHealth:
    """Health status of a node."""
    
    node_id: str
    
    # Status
    status: HealthStatus = HealthStatus.UNKNOWN
    status_message: str = ""
    
    # Metrics
    metrics: HealthMetrics = field(default_factory=HealthMetrics)
    
    # History
    status_history: List[tuple] = field(default_factory=list)  # (timestamp, status)
    
    # Timestamp
    last_checked: float = 0.0


@dataclass
class HealthAlert:
    """A health alert."""
    
    alert_id: str
    node_id: str
    
    # Alert info
    severity: AlertSeverity
    message: str
    
    # Context
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    
    # Timing
    created_at: float = field(default_factory=time.time)
    acknowledged: bool = False
    acknowledged_at: Optional[float] = None
    
    # Resolution
    resolved: bool = False
    resolved_at: Optional[float] = None


@dataclass
class HealthThresholds:
    """Thresholds for health checks."""
    
    # Heartbeat
    heartbeat_warning_s: float = 60.0
    heartbeat_critical_s: float = 120.0
    
    # Performance
    latency_warning_pct: float = 20.0   # 20% above baseline
    latency_critical_pct: float = 50.0
    
    # Success rate
    success_warning_pct: float = 95.0
    success_critical_pct: float = 90.0
    
    # Temperature
    temp_warning_c: float = 80.0
    temp_critical_c: float = 90.0
    
    # Memory
    memory_warning_pct: float = 85.0
    memory_critical_pct: float = 95.0


class HealthMonitor:
    """
    Monitors health of fleet nodes.
    
    Features:
    - Periodic health checks
    - Threshold-based alerting
    - Historical tracking
    - Alert management
    """
    
    def __init__(self, fleet_manager: FleetManager,
                 thresholds: Optional[HealthThresholds] = None):
        self.fleet = fleet_manager
        self.thresholds = thresholds or HealthThresholds()
        
        self._node_health: Dict[str, NodeHealth] = {}
        self._alerts: Dict[str, HealthAlert] = {}
        self._alert_counter = 0
        
        # Callbacks
        self._alert_callbacks: List[Callable] = []
        
        # Baseline performance
        self._baselines: Dict[str, float] = {}  # node_id -> baseline_latency
    
    def check_all(self) -> Dict[str, NodeHealth]:
        """
        Check health of all nodes.
        
        Returns:
            Dict mapping node_id to health status
        """
        for node in self.fleet.get_all_nodes():
            self._node_health[node.node_id] = self.check_node(node)
        
        return self._node_health
    
    def check_node(self, node: GPUNode) -> NodeHealth:
        """
        Check health of a single node.
        """
        health = self._node_health.get(node.node_id, NodeHealth(node_id=node.node_id))
        health.last_checked = time.time()
        
        issues = []
        
        # Check heartbeat
        heartbeat_delay = time.time() - node.last_seen
        health.metrics.last_heartbeat_delay_s = heartbeat_delay
        
        if heartbeat_delay > self.thresholds.heartbeat_critical_s:
            issues.append(("critical", "Heartbeat timeout"))
            self._create_alert(node.node_id, AlertSeverity.CRITICAL,
                             f"Node heartbeat timeout: {heartbeat_delay:.0f}s")
        elif heartbeat_delay > self.thresholds.heartbeat_warning_s:
            issues.append(("warning", "Heartbeat delayed"))
            self._create_alert(node.node_id, AlertSeverity.WARNING,
                             f"Node heartbeat delayed: {heartbeat_delay:.0f}s")
        
        # Check node status
        if node.status == NodeStatus.ERROR:
            issues.append(("critical", "Node in error state"))
        elif node.status == NodeStatus.MAINTENANCE:
            issues.append(("warning", "Node under maintenance"))
        elif node.status == NodeStatus.OFFLINE:
            issues.append(("critical", "Node offline"))
        
        # Determine overall status
        critical_issues = sum(1 for sev, _ in issues if sev == "critical")
        warning_issues = sum(1 for sev, _ in issues if sev == "warning")
        
        if critical_issues > 0:
            health.status = HealthStatus.CRITICAL
        elif warning_issues > 0:
            health.status = HealthStatus.WARNING
        else:
            health.status = HealthStatus.HEALTHY
        
        # Status message
        if issues:
            health.status_message = "; ".join(msg for _, msg in issues)
        else:
            health.status_message = "All checks passed"
        
        # Update history
        health.status_history.append((time.time(), health.status))
        # Keep last 100 entries
        if len(health.status_history) > 100:
            health.status_history = health.status_history[-100:]
        
        return health
    
    def update_metrics(self, node_id: str, metrics: Dict[str, float]) -> None:
        """
        Update metrics for a node.
        
        Called with data from the node agent.
        """
        if node_id not in self._node_health:
            self._node_health[node_id] = NodeHealth(node_id=node_id)
        
        health = self._node_health[node_id]
        
        # Update metrics
        if "gpu_utilization" in metrics:
            health.metrics.gpu_utilization = metrics["gpu_utilization"]
        if "memory_utilization" in metrics:
            health.metrics.memory_utilization = metrics["memory_utilization"]
            # Check threshold
            if metrics["memory_utilization"] > self.thresholds.memory_critical_pct:
                self._create_alert(node_id, AlertSeverity.CRITICAL,
                                 f"Memory usage critical: {metrics['memory_utilization']:.1f}%")
            elif metrics["memory_utilization"] > self.thresholds.memory_warning_pct:
                self._create_alert(node_id, AlertSeverity.WARNING,
                                 f"Memory usage high: {metrics['memory_utilization']:.1f}%")
        
        if "temperature_c" in metrics:
            health.metrics.temperature_c = metrics["temperature_c"]
            if metrics["temperature_c"] > self.thresholds.temp_critical_c:
                self._create_alert(node_id, AlertSeverity.CRITICAL,
                                 f"GPU temperature critical: {metrics['temperature_c']:.1f}°C")
            elif metrics["temperature_c"] > self.thresholds.temp_warning_c:
                self._create_alert(node_id, AlertSeverity.WARNING,
                                 f"GPU temperature high: {metrics['temperature_c']:.1f}°C")
        
        if "power_watts" in metrics:
            health.metrics.power_watts = metrics["power_watts"]
    
    def record_job_result(self, node_id: str, success: bool,
                         latency_ms: float) -> None:
        """
        Record a job result for health tracking.
        """
        if node_id not in self._node_health:
            self._node_health[node_id] = NodeHealth(node_id=node_id)
        
        health = self._node_health[node_id]
        
        if not success:
            health.metrics.error_count += 1
        
        # Check latency vs baseline
        if node_id in self._baselines and self._baselines[node_id] > 0:
            baseline = self._baselines[node_id]
            ratio = latency_ms / baseline
            health.metrics.avg_latency_vs_baseline = ratio
            
            degradation_pct = (ratio - 1) * 100
            if degradation_pct > self.thresholds.latency_critical_pct:
                self._create_alert(node_id, AlertSeverity.ERROR,
                                 f"Performance degraded: {degradation_pct:.1f}% slower than baseline")
            elif degradation_pct > self.thresholds.latency_warning_pct:
                self._create_alert(node_id, AlertSeverity.WARNING,
                                 f"Performance degraded: {degradation_pct:.1f}% slower than baseline")
    
    def set_baseline(self, node_id: str, baseline_latency_ms: float) -> None:
        """Set baseline latency for a node."""
        self._baselines[node_id] = baseline_latency_ms
    
    def _create_alert(self, node_id: str, severity: AlertSeverity,
                     message: str) -> HealthAlert:
        """Create and store an alert."""
        self._alert_counter += 1
        
        alert = HealthAlert(
            alert_id=f"alert_{self._alert_counter:05d}",
            node_id=node_id,
            severity=severity,
            message=message,
        )
        
        self._alerts[alert.alert_id] = alert
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception:
                pass
        
        return alert
    
    def get_node_health(self, node_id: str) -> Optional[NodeHealth]:
        """Get health for a node."""
        return self._node_health.get(node_id)
    
    def get_active_alerts(self) -> List[HealthAlert]:
        """Get all unresolved alerts."""
        return [a for a in self._alerts.values() if not a.resolved]
    
    def get_alerts_by_node(self, node_id: str) -> List[HealthAlert]:
        """Get alerts for a specific node."""
        return [a for a in self._alerts.values() if a.node_id == node_id]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].acknowledged = True
            self._alerts[alert_id].acknowledged_at = time.time()
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].resolved = True
            self._alerts[alert_id].resolved_at = time.time()
            return True
        return False
    
    def get_fleet_health_summary(self) -> Dict[str, Any]:
        """
        Get summary of fleet health.
        """
        nodes = list(self._node_health.values())
        
        return {
            "total_nodes": len(nodes),
            "healthy_nodes": sum(1 for n in nodes if n.status == HealthStatus.HEALTHY),
            "warning_nodes": sum(1 for n in nodes if n.status == HealthStatus.WARNING),
            "degraded_nodes": sum(1 for n in nodes if n.status == HealthStatus.DEGRADED),
            "critical_nodes": sum(1 for n in nodes if n.status == HealthStatus.CRITICAL),
            "active_alerts": len(self.get_active_alerts()),
            "unacknowledged_alerts": sum(1 for a in self._alerts.values() 
                                        if not a.resolved and not a.acknowledged),
        }
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for new alerts."""
        self._alert_callbacks.append(callback)
