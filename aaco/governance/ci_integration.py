"""
AACO-SIGMA CI/CD Integration

Integrates performance governance into CI/CD pipelines.
Provides quality gates and automated performance validation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum, auto
import time
import json
from pathlib import Path

from .regression_detector import RegressionDetector, RegressionResult, RegressionSeverity
from .sla_engine import SLAEngine, SLAResult, SLAMetricType
from .baseline_manager import BaselineManager, BaselineComparison


class GateDecision(Enum):
    """CI gate decision."""
    PASS = auto()      # Allow pipeline to continue
    WARN = auto()      # Continue with warnings
    FAIL = auto()      # Block pipeline
    SKIP = auto()      # Gate not applicable


class GateType(Enum):
    """Types of quality gates."""
    REGRESSION = auto()      # Regression detection gate
    SLA = auto()             # SLA compliance gate
    BASELINE = auto()        # Baseline comparison gate
    CUSTOM = auto()          # Custom gate


@dataclass
class GateResult:
    """Result of a single gate evaluation."""
    gate_name: str
    gate_type: GateType
    decision: GateDecision
    
    # Details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    evaluation_time_ms: float = 0.0


@dataclass
class PipelineConfig:
    """Configuration for CI pipeline integration."""
    
    # Pipeline identity
    pipeline_name: str = ""
    pipeline_id: str = ""
    
    # Gate configuration
    enable_regression_gate: bool = True
    enable_sla_gate: bool = True
    enable_baseline_gate: bool = True
    
    # Thresholds
    max_regression_severity: RegressionSeverity = RegressionSeverity.MODERATE
    sla_policy_name: str = "production_default"
    baseline_name: str = ""
    
    # Behavior
    fail_fast: bool = False  # Stop on first failure
    allow_warnings: bool = True  # Warnings don't block
    
    # Output
    output_format: str = "json"  # json, junit, markdown
    output_path: Optional[Path] = None


@dataclass
class CIPipelineResult:
    """Complete result of CI pipeline evaluation."""
    
    # Identity
    pipeline_name: str
    run_id: str
    
    # Overall decision
    decision: GateDecision
    
    # Gate results
    gates: List[GateResult] = field(default_factory=list)
    
    # Summary
    passed_gates: int = 0
    warned_gates: int = 0
    failed_gates: int = 0
    skipped_gates: int = 0
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_ms: float = 0.0
    
    # Artifacts
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    git_commit: str = ""
    git_branch: str = ""


class CIIntegration:
    """
    CI/CD pipeline integration for performance governance.
    
    Provides:
    - Quality gates for performance
    - JUnit/JSON output for CI systems
    - GitHub Actions / GitLab CI integration
    - Slack/Teams notifications
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.regression_detector = RegressionDetector()
        self.sla_engine = SLAEngine()
        self.baseline_manager = BaselineManager()
        
        # Hooks
        self._pre_hooks: List[Callable] = []
        self._post_hooks: List[Callable] = []
    
    def run_gates(self,
                  metrics: Dict[str, Any],
                  baseline_values: Optional[Dict[str, List[float]]] = None,
                  run_id: str = "") -> CIPipelineResult:
        """
        Run all configured quality gates.
        
        Args:
            metrics: Current performance metrics
            baseline_values: Historical values for regression detection
            run_id: Unique run identifier
        """
        import uuid
        
        if not run_id:
            run_id = str(uuid.uuid4())[:8]
        
        result = CIPipelineResult(
            pipeline_name=self.config.pipeline_name,
            run_id=run_id,
            decision=GateDecision.PASS,
            start_time=time.time(),
        )
        
        # Run pre-hooks
        for hook in self._pre_hooks:
            hook(result)
        
        try:
            # Regression gate
            if self.config.enable_regression_gate and baseline_values:
                gate_result = self._run_regression_gate(metrics, baseline_values)
                result.gates.append(gate_result)
                
                if gate_result.decision == GateDecision.FAIL and self.config.fail_fast:
                    result.decision = GateDecision.FAIL
                    return self._finalize_result(result)
            
            # SLA gate
            if self.config.enable_sla_gate:
                gate_result = self._run_sla_gate(metrics)
                result.gates.append(gate_result)
                
                if gate_result.decision == GateDecision.FAIL and self.config.fail_fast:
                    result.decision = GateDecision.FAIL
                    return self._finalize_result(result)
            
            # Baseline gate
            if self.config.enable_baseline_gate and self.config.baseline_name:
                gate_result = self._run_baseline_gate(metrics)
                result.gates.append(gate_result)
            
            # Aggregate decision
            result = self._finalize_result(result)
            
        finally:
            # Run post-hooks
            for hook in self._post_hooks:
                hook(result)
        
        return result
    
    def _run_regression_gate(self,
                             metrics: Dict[str, Any],
                             baseline_values: Dict[str, List[float]]) -> GateResult:
        """Run regression detection gate."""
        start = time.time()
        
        regressions: List[RegressionResult] = []
        
        for metric_name, current_value in metrics.items():
            if metric_name in baseline_values:
                baseline = baseline_values[metric_name]
                current = [current_value] if isinstance(current_value, (int, float)) else current_value
                
                reg_result = self.regression_detector.detect(
                    metric_name, baseline, current
                )
                
                if reg_result.is_regression:
                    regressions.append(reg_result)
        
        # Determine decision based on worst regression
        decision = GateDecision.PASS
        worst_severity = RegressionSeverity.NONE
        
        for reg in regressions:
            if reg.severity.value > worst_severity.value:
                worst_severity = reg.severity
        
        if worst_severity.value > self.config.max_regression_severity.value:
            decision = GateDecision.FAIL
        elif worst_severity != RegressionSeverity.NONE:
            decision = GateDecision.WARN
        
        return GateResult(
            gate_name="regression_detection",
            gate_type=GateType.REGRESSION,
            decision=decision,
            message=f"Found {len(regressions)} regressions, worst: {worst_severity.name}",
            details={
                "regressions": [
                    {
                        "metric": r.metric_name,
                        "delta_pct": r.delta_pct,
                        "severity": r.severity.name,
                    }
                    for r in regressions
                ],
                "worst_severity": worst_severity.name,
            },
            evaluation_time_ms=(time.time() - start) * 1000,
        )
    
    def _run_sla_gate(self, metrics: Dict[str, Any]) -> GateResult:
        """Run SLA compliance gate."""
        start = time.time()
        
        # Convert metrics to SLA format
        sla_metrics: Dict[SLAMetricType, float] = {}
        
        metric_type_map = {
            "latency_p50": SLAMetricType.LATENCY_P50,
            "latency_p95": SLAMetricType.LATENCY_P95,
            "latency_p99": SLAMetricType.LATENCY_P99,
            "throughput": SLAMetricType.THROUGHPUT,
            "memory_peak": SLAMetricType.MEMORY_PEAK,
            "gpu_utilization": SLAMetricType.GPU_UTILIZATION,
        }
        
        for key, metric_type in metric_type_map.items():
            if key in metrics:
                sla_metrics[metric_type] = metrics[key]
        
        # Ensure policy exists
        if self.config.sla_policy_name not in self.sla_engine.policies:
            self.sla_engine.add_policy(SLAEngine.DEFAULT_PRODUCTION_SLA)
        
        sla_result = self.sla_engine.evaluate(
            sla_metrics,
            policy_name=self.config.sla_policy_name,
        )
        
        decision = GateDecision.PASS if sla_result.passed else GateDecision.FAIL
        if sla_result.passed and sla_result.warnings:
            decision = GateDecision.WARN
        
        return GateResult(
            gate_name="sla_compliance",
            gate_type=GateType.SLA,
            decision=decision,
            message=f"SLA check: {sla_result.passed_checks}/{sla_result.total_checks} passed",
            details={
                "violations": [
                    {"metric": v.metric_name, "actual": v.actual_value, "threshold": v.threshold}
                    for v in sla_result.violations
                ],
                "warnings": [
                    {"metric": w.metric_name, "margin_pct": w.margin_pct}
                    for w in sla_result.warnings
                ],
            },
            evaluation_time_ms=(time.time() - start) * 1000,
        )
    
    def _run_baseline_gate(self, metrics: Dict[str, Any]) -> GateResult:
        """Run baseline comparison gate."""
        start = time.time()
        
        baseline = self.baseline_manager.get_active_baseline(self.config.baseline_name)
        
        if baseline is None:
            return GateResult(
                gate_name="baseline_comparison",
                gate_type=GateType.BASELINE,
                decision=GateDecision.SKIP,
                message=f"Baseline '{self.config.baseline_name}' not found",
                evaluation_time_ms=(time.time() - start) * 1000,
            )
        
        # Compare metrics
        regressions = []
        improvements = []
        
        for metric_name, current_value in metrics.items():
            baseline_metric = baseline.get_metric(metric_name)
            if baseline_metric:
                delta_pct = ((current_value - baseline_metric.mean) / baseline_metric.mean * 100) if baseline_metric.mean != 0 else 0
                
                if delta_pct > 5:  # 5% regression
                    regressions.append((metric_name, delta_pct))
                elif delta_pct < -5:  # 5% improvement
                    improvements.append((metric_name, delta_pct))
        
        decision = GateDecision.PASS
        if regressions:
            decision = GateDecision.WARN
            if any(d > 15 for _, d in regressions):  # Major regression
                decision = GateDecision.FAIL
        
        return GateResult(
            gate_name="baseline_comparison",
            gate_type=GateType.BASELINE,
            decision=decision,
            message=f"Compared to baseline: {len(regressions)} regressions, {len(improvements)} improvements",
            details={
                "baseline_version": baseline.version.version,
                "regressions": [{"metric": m, "delta_pct": d} for m, d in regressions],
                "improvements": [{"metric": m, "delta_pct": d} for m, d in improvements],
            },
            evaluation_time_ms=(time.time() - start) * 1000,
        )
    
    def _finalize_result(self, result: CIPipelineResult) -> CIPipelineResult:
        """Finalize and aggregate results."""
        result.end_time = time.time()
        result.total_duration_ms = (result.end_time - result.start_time) * 1000
        
        # Count results
        for gate in result.gates:
            if gate.decision == GateDecision.PASS:
                result.passed_gates += 1
            elif gate.decision == GateDecision.WARN:
                result.warned_gates += 1
            elif gate.decision == GateDecision.FAIL:
                result.failed_gates += 1
            else:
                result.skipped_gates += 1
        
        # Aggregate decision
        if result.failed_gates > 0:
            result.decision = GateDecision.FAIL
        elif result.warned_gates > 0 and not self.config.allow_warnings:
            result.decision = GateDecision.FAIL
        elif result.warned_gates > 0:
            result.decision = GateDecision.WARN
        else:
            result.decision = GateDecision.PASS
        
        # Generate output
        if self.config.output_path:
            self._write_output(result)
        
        return result
    
    def _write_output(self, result: CIPipelineResult) -> None:
        """Write result to output file."""
        if self.config.output_format == "json":
            self._write_json(result)
        elif self.config.output_format == "junit":
            self._write_junit(result)
        elif self.config.output_format == "markdown":
            self._write_markdown(result)
    
    def _write_json(self, result: CIPipelineResult) -> None:
        """Write JSON output."""
        output = {
            "pipeline": result.pipeline_name,
            "run_id": result.run_id,
            "decision": result.decision.name,
            "duration_ms": result.total_duration_ms,
            "gates": [
                {
                    "name": g.gate_name,
                    "decision": g.decision.name,
                    "message": g.message,
                    "details": g.details,
                }
                for g in result.gates
            ],
            "summary": {
                "passed": result.passed_gates,
                "warned": result.warned_gates,
                "failed": result.failed_gates,
                "skipped": result.skipped_gates,
            },
        }
        
        if self.config.output_path:
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.output_path, 'w') as f:
                json.dump(output, f, indent=2)
    
    def _write_junit(self, result: CIPipelineResult) -> None:
        """Write JUnit XML output for CI systems."""
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="{result.pipeline_name}" tests="{len(result.gates)}" '
            f'failures="{result.failed_gates}" time="{result.total_duration_ms / 1000:.3f}">',
        ]
        
        for gate in result.gates:
            status = ""
            if gate.decision == GateDecision.FAIL:
                status = f'<failure message="{gate.message}"/>'
            elif gate.decision == GateDecision.SKIP:
                status = "<skipped/>"
            
            xml_lines.append(
                f'  <testcase name="{gate.gate_name}" '
                f'time="{gate.evaluation_time_ms / 1000:.3f}">'
            )
            if status:
                xml_lines.append(f'    {status}')
            xml_lines.append('  </testcase>')
        
        xml_lines.append('</testsuite>')
        
        if self.config.output_path:
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.output_path, 'w') as f:
                f.write('\n'.join(xml_lines))
    
    def _write_markdown(self, result: CIPipelineResult) -> None:
        """Write Markdown output for PRs."""
        icon = "✅" if result.decision == GateDecision.PASS else "❌" if result.decision == GateDecision.FAIL else "⚠️"
        
        lines = [
            f"# {icon} Performance Gate Results",
            "",
            f"**Pipeline:** {result.pipeline_name}",
            f"**Run ID:** {result.run_id}",
            f"**Duration:** {result.total_duration_ms:.1f}ms",
            "",
            "## Gate Results",
            "",
            "| Gate | Decision | Message |",
            "|------|----------|---------|",
        ]
        
        for gate in result.gates:
            icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(gate.decision.name, "")
            lines.append(f"| {gate.gate_name} | {icon} {gate.decision.name} | {gate.message} |")
        
        lines.extend([
            "",
            f"**Summary:** {result.passed_gates} passed, {result.warned_gates} warnings, {result.failed_gates} failed",
        ])
        
        if self.config.output_path:
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.output_path, 'w') as f:
                f.write('\n'.join(lines))
    
    def register_pre_hook(self, hook: Callable) -> None:
        """Register a pre-evaluation hook."""
        self._pre_hooks.append(hook)
    
    def register_post_hook(self, hook: Callable) -> None:
        """Register a post-evaluation hook."""
        self._post_hooks.append(hook)
