"""
AACO-SIGMA SLA Engine

Service Level Agreement policies and enforcement.
Defines, monitors, and enforces performance SLAs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum, auto
import time


class SLAMetricType(Enum):
    """Types of SLA metrics."""
    LATENCY_P50 = auto()      # 50th percentile latency
    LATENCY_P95 = auto()      # 95th percentile latency
    LATENCY_P99 = auto()      # 99th percentile latency
    THROUGHPUT = auto()       # Tokens/second, samples/second
    MEMORY_PEAK = auto()      # Peak memory usage
    GPU_UTILIZATION = auto()  # GPU utilization percentage
    POWER_CONSUMPTION = auto() # Power in watts
    CUSTOM = auto()           # Custom metric


class ViolationSeverity(Enum):
    """Severity of SLA violations."""
    WARNING = auto()    # Close to threshold
    VIOLATION = auto()  # Exceeded threshold
    CRITICAL = auto()   # Severely exceeded


@dataclass
class SLACheck:
    """A single SLA check result."""
    metric_name: str
    metric_type: SLAMetricType
    
    # Values
    threshold: float
    actual_value: float
    
    # Result
    passed: bool
    severity: Optional[ViolationSeverity] = None
    margin_pct: float = 0.0  # How much margin/headroom
    
    # Context
    timestamp: float = field(default_factory=time.time)
    details: str = ""


@dataclass
class SLAViolation:
    """Record of an SLA violation."""
    metric_name: str
    metric_type: SLAMetricType
    policy_name: str
    
    # Violation details
    threshold: float
    actual_value: float
    overage_pct: float
    severity: ViolationSeverity
    
    # Timing
    timestamp: float
    duration_s: float = 0.0  # How long violation persisted
    
    # Context
    environment: str = ""
    model_name: str = ""
    kernel_name: str = ""
    
    # Metadata
    acknowledged: bool = False
    resolution: str = ""


@dataclass
class SLAPolicy:
    """Definition of an SLA policy."""
    name: str
    description: str = ""
    
    # Thresholds
    thresholds: Dict[SLAMetricType, float] = field(default_factory=dict)
    
    # Warning thresholds (percentage of threshold)
    warning_pct: float = 90.0  # Warn at 90% of threshold
    
    # Enforcement
    enabled: bool = True
    blocking: bool = False  # Block CI on violation?
    
    # Scope
    applies_to_models: List[str] = field(default_factory=list)
    applies_to_envs: List[str] = field(default_factory=list)
    
    # Validity
    valid_from: float = 0.0
    valid_until: float = float('inf')


@dataclass
class SLAResult:
    """Result of SLA evaluation."""
    policy_name: str
    passed: bool
    
    # Individual checks
    checks: List[SLACheck] = field(default_factory=list)
    
    # Violations
    violations: List[SLAViolation] = field(default_factory=list)
    warnings: List[SLACheck] = field(default_factory=list)
    
    # Summary
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    
    # Timing
    evaluation_time_ms: float = 0.0


class SLAEngine:
    """
    SLA definition, monitoring, and enforcement engine.
    
    Supports:
    - Multiple SLA policies
    - Various metric types
    - Warning thresholds
    - Violation tracking and history
    """
    
    # Default production SLA policy
    DEFAULT_PRODUCTION_SLA = SLAPolicy(
        name="production_default",
        description="Default production SLA for AI inference",
        thresholds={
            SLAMetricType.LATENCY_P50: 50.0,      # 50ms
            SLAMetricType.LATENCY_P95: 100.0,     # 100ms
            SLAMetricType.LATENCY_P99: 200.0,     # 200ms
            SLAMetricType.THROUGHPUT: 100.0,      # 100 tok/s
            SLAMetricType.MEMORY_PEAK: 16000.0,   # 16GB
            SLAMetricType.GPU_UTILIZATION: 70.0,  # 70% min
        },
        warning_pct=85.0,
        blocking=True,
    )
    
    def __init__(self):
        self.policies: Dict[str, SLAPolicy] = {}
        self.violations: List[SLAViolation] = []
        self._hooks: List[Callable[[SLAViolation], None]] = []
    
    def add_policy(self, policy: SLAPolicy) -> None:
        """Add an SLA policy."""
        self.policies[policy.name] = policy
    
    def remove_policy(self, name: str) -> bool:
        """Remove an SLA policy."""
        if name in self.policies:
            del self.policies[name]
            return True
        return False
    
    def get_policy(self, name: str) -> Optional[SLAPolicy]:
        """Get an SLA policy by name."""
        return self.policies.get(name)
    
    def evaluate(self, 
                 metrics: Dict[SLAMetricType, float],
                 policy_name: Optional[str] = None,
                 context: Optional[Dict[str, str]] = None) -> SLAResult:
        """
        Evaluate metrics against SLA policy.
        
        Args:
            metrics: Current metric values
            policy_name: Specific policy to check (default: all)
            context: Additional context (model_name, environment, etc.)
        """
        start_time = time.time()
        context = context or {}
        
        if policy_name:
            policies_to_check = [self.policies[policy_name]] if policy_name in self.policies else []
        else:
            policies_to_check = list(self.policies.values())
        
        # Aggregate results
        all_checks: List[SLACheck] = []
        all_violations: List[SLAViolation] = []
        all_warnings: List[SLACheck] = []
        
        for policy in policies_to_check:
            if not policy.enabled:
                continue
            
            # Check if policy applies to this context
            if policy.applies_to_models and context.get("model_name") not in policy.applies_to_models:
                continue
            if policy.applies_to_envs and context.get("environment") not in policy.applies_to_envs:
                continue
            
            for metric_type, threshold in policy.thresholds.items():
                if metric_type not in metrics:
                    continue
                
                actual = metrics[metric_type]
                
                # Create check
                check = self._check_metric(
                    metric_type, actual, threshold, policy.warning_pct
                )
                all_checks.append(check)
                
                # Handle failures
                if not check.passed:
                    violation = SLAViolation(
                        metric_name=check.metric_name,
                        metric_type=metric_type,
                        policy_name=policy.name,
                        threshold=threshold,
                        actual_value=actual,
                        overage_pct=((actual - threshold) / threshold * 100) if threshold > 0 else 0,
                        severity=check.severity or ViolationSeverity.VIOLATION,
                        timestamp=time.time(),
                        environment=context.get("environment", ""),
                        model_name=context.get("model_name", ""),
                        kernel_name=context.get("kernel_name", ""),
                    )
                    all_violations.append(violation)
                    self.violations.append(violation)
                    
                    # Notify hooks
                    for hook in self._hooks:
                        hook(violation)
                
                elif check.severity == ViolationSeverity.WARNING:
                    all_warnings.append(check)
        
        # Build result
        result = SLAResult(
            policy_name=policy_name or "all",
            passed=len(all_violations) == 0,
            checks=all_checks,
            violations=all_violations,
            warnings=all_warnings,
            total_checks=len(all_checks),
            passed_checks=sum(1 for c in all_checks if c.passed),
            failed_checks=len(all_violations),
            warning_checks=len(all_warnings),
            evaluation_time_ms=(time.time() - start_time) * 1000,
        )
        
        return result
    
    def _check_metric(self,
                      metric_type: SLAMetricType,
                      actual: float,
                      threshold: float,
                      warning_pct: float) -> SLACheck:
        """Check a single metric against threshold."""
        check = SLACheck(
            metric_name=metric_type.name,
            metric_type=metric_type,
            threshold=threshold,
            actual_value=actual,
            passed=True,
        )
        
        # Different metric types have different comparison logic
        if metric_type in [SLAMetricType.GPU_UTILIZATION, SLAMetricType.THROUGHPUT]:
            # Higher is better - fail if below threshold
            check.passed = actual >= threshold
            check.margin_pct = ((actual - threshold) / threshold * 100) if threshold > 0 else 0
            
            warning_threshold = threshold * (warning_pct / 100)
            if actual < threshold:
                check.severity = ViolationSeverity.VIOLATION
            elif actual < threshold * 1.1:  # Within 10% of threshold
                check.severity = ViolationSeverity.WARNING
        else:
            # Lower is better (latency, memory) - fail if above threshold
            check.passed = actual <= threshold
            check.margin_pct = ((threshold - actual) / threshold * 100) if threshold > 0 else 0
            
            warning_threshold = threshold * (warning_pct / 100)
            if actual > threshold:
                if actual > threshold * 1.5:
                    check.severity = ViolationSeverity.CRITICAL
                else:
                    check.severity = ViolationSeverity.VIOLATION
            elif actual > warning_threshold:
                check.severity = ViolationSeverity.WARNING
        
        return check
    
    def register_violation_hook(self, hook: Callable[[SLAViolation], None]) -> None:
        """Register a callback for violations."""
        self._hooks.append(hook)
    
    def get_violations(self,
                       since: Optional[float] = None,
                       policy_name: Optional[str] = None,
                       severity: Optional[ViolationSeverity] = None) -> List[SLAViolation]:
        """Get filtered violations."""
        violations = self.violations
        
        if since:
            violations = [v for v in violations if v.timestamp >= since]
        
        if policy_name:
            violations = [v for v in violations if v.policy_name == policy_name]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        return violations
    
    def acknowledge_violation(self, violation: SLAViolation, resolution: str = "") -> None:
        """Mark a violation as acknowledged."""
        violation.acknowledged = True
        violation.resolution = resolution
    
    def generate_report(self, result: SLAResult) -> Dict[str, Any]:
        """Generate an SLA report."""
        return {
            "policy": result.policy_name,
            "status": "PASS" if result.passed else "FAIL",
            "checks": {
                "total": result.total_checks,
                "passed": result.passed_checks,
                "failed": result.failed_checks,
                "warnings": result.warning_checks,
            },
            "violations": [
                {
                    "metric": v.metric_name,
                    "threshold": v.threshold,
                    "actual": v.actual_value,
                    "overage_pct": v.overage_pct,
                    "severity": v.severity.name,
                }
                for v in result.violations
            ],
            "warnings": [
                {
                    "metric": w.metric_name,
                    "threshold": w.threshold,
                    "actual": w.actual_value,
                    "margin_pct": w.margin_pct,
                }
                for w in result.warnings
            ],
            "evaluation_time_ms": result.evaluation_time_ms,
        }
