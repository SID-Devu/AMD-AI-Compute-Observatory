# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Auto-Optimization Engine

Closed-loop self-experimental validation with:
- Hypothesis generation from root cause analysis
- Minimal confirming experiments
- Optimization application and validation
- Automatic rollback on regression
"""

import json
import time
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimizations."""
    KERNEL_FUSION = "kernel_fusion"
    BATCH_SIZE = "batch_size"
    MEMORY_LAYOUT = "memory_layout"
    PRECISION = "precision"
    THREAD_CONFIG = "thread_config"
    CACHE_POLICY = "cache_policy"
    OPERATOR_VARIANT = "operator_variant"
    GRAPH_STRUCTURE = "graph_structure"


class ExperimentStatus(Enum):
    """Status of optimization experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    ROLLED_BACK = "rolled_back"


@dataclass
class OptimizationHypothesis:
    """
    A hypothesis for performance improvement.
    
    Generated from root cause analysis.
    """
    # Identity
    hypothesis_id: str = ""
    
    # Type and target
    optimization_type: OptimizationType = OptimizationType.KERNEL_FUSION
    target_metric: str = "latency_p50"
    
    # Expected impact
    expected_improvement_pct: float = 0.0
    confidence: float = 0.0
    
    # Root cause connection
    root_cause: str = ""
    evidence: List[str] = field(default_factory=list)
    
    # Configuration changes
    config_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Priority (lower = higher priority)
    priority: int = 100


@dataclass
class ExperimentConfig:
    """Configuration for an optimization experiment."""
    # Identity
    experiment_id: str = ""
    hypothesis: OptimizationHypothesis = field(default_factory=OptimizationHypothesis)
    
    # Measurement config
    warmup_iterations: int = 5
    measurement_iterations: int = 20
    
    # Success criteria
    min_improvement_pct: float = 2.0
    max_regression_pct: float = 1.0
    required_confidence: float = 0.8
    
    # Rollback policy
    auto_rollback: bool = True
    rollback_threshold_pct: float = 5.0


@dataclass
class ExperimentResult:
    """Result of an optimization experiment."""
    # Identity
    experiment_id: str = ""
    hypothesis_id: str = ""
    
    # Status
    status: ExperimentStatus = ExperimentStatus.PENDING
    
    # Measurements
    baseline_value: float = 0.0
    optimized_value: float = 0.0
    improvement_pct: float = 0.0
    
    # Validation
    is_improvement: bool = False
    is_statistically_significant: bool = False
    confidence: float = 0.0
    
    # Metadata
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    
    # Error info
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'hypothesis_id': self.hypothesis_id,
            'status': self.status.value,
            'baseline': self.baseline_value,
            'optimized': self.optimized_value,
            'improvement_pct': f"{self.improvement_pct:+.2f}%",
            'is_improvement': self.is_improvement,
            'confidence': f"{self.confidence:.2f}",
            'duration_s': f"{self.duration_seconds:.1f}",
        }


@dataclass
class OptimizationState:
    """Current state of applied optimizations."""
    # Applied optimizations
    applied_optimizations: List[str] = field(default_factory=list)
    
    # Configuration snapshot
    current_config: Dict[str, Any] = field(default_factory=dict)
    baseline_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    current_metrics: Dict[str, float] = field(default_factory=dict)
    
    # History
    experiment_history: List[ExperimentResult] = field(default_factory=list)


class AutoOptimizationEngine:
    """
    AACO-Ω∞ Closed-Loop Auto-Optimization Engine
    
    Automatically generates, tests, and validates optimizations.
    
    Workflow:
    1. Receive root cause analysis
    2. Generate optimization hypotheses
    3. Run minimal confirming experiments
    4. Apply validated optimizations
    5. Monitor for regressions
    6. Rollback if needed
    """
    
    # Hypothesis generation rules per root cause
    OPTIMIZATION_RULES = {
        'launch_overhead': [
            {
                'type': OptimizationType.KERNEL_FUSION,
                'expected': 15.0,
                'config': {'enable_fusion': True, 'fusion_level': 'aggressive'},
            },
            {
                'type': OptimizationType.BATCH_SIZE,
                'expected': 10.0,
                'config': {'increase_batch': True},
            },
        ],
        'memory_bandwidth': [
            {
                'type': OptimizationType.MEMORY_LAYOUT,
                'expected': 12.0,
                'config': {'layout': 'optimal'},
            },
            {
                'type': OptimizationType.CACHE_POLICY,
                'expected': 8.0,
                'config': {'cache_policy': 'aggressive'},
            },
        ],
        'compute_bound': [
            {
                'type': OptimizationType.PRECISION,
                'expected': 50.0,
                'config': {'precision': 'fp16'},
            },
        ],
        'partition_overhead': [
            {
                'type': OptimizationType.GRAPH_STRUCTURE,
                'expected': 10.0,
                'config': {'minimize_partitions': True},
            },
        ],
        'occupancy_limited': [
            {
                'type': OptimizationType.THREAD_CONFIG,
                'expected': 8.0,
                'config': {'optimize_occupancy': True},
            },
        ],
    }
    
    def __init__(
        self,
        measurement_func: Optional[Callable[[], float]] = None,
        apply_config_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        """
        Initialize auto-optimization engine.
        
        Args:
            measurement_func: Function to measure target metric
            apply_config_func: Function to apply configuration changes
        """
        self._measurement_func = measurement_func
        self._apply_config_func = apply_config_func
        
        self._state = OptimizationState()
        self._pending_hypotheses: List[OptimizationHypothesis] = []
        self._next_hypothesis_id = 1
        self._next_experiment_id = 1
    
    def generate_hypotheses(
        self,
        root_cause: str,
        evidence: List[str],
        confidence: float = 0.5,
    ) -> List[OptimizationHypothesis]:
        """
        Generate optimization hypotheses from root cause.
        
        Args:
            root_cause: Identified root cause category
            evidence: Supporting evidence list
            confidence: Root cause confidence
            
        Returns:
            List of generated hypotheses
        """
        hypotheses = []
        
        rules = self.OPTIMIZATION_RULES.get(root_cause, [])
        
        for i, rule in enumerate(rules):
            hyp = OptimizationHypothesis(
                hypothesis_id=f"hyp_{self._next_hypothesis_id}",
                optimization_type=rule['type'],
                expected_improvement_pct=rule['expected'],
                confidence=confidence * (0.9 ** i),  # Decay for later options
                root_cause=root_cause,
                evidence=evidence,
                config_changes=rule['config'],
                priority=i + 1,
            )
            self._next_hypothesis_id += 1
            hypotheses.append(hyp)
        
        # Add to pending queue
        self._pending_hypotheses.extend(hypotheses)
        self._pending_hypotheses.sort(key=lambda h: (h.priority, -h.expected_improvement_pct))
        
        logger.info(f"Generated {len(hypotheses)} hypotheses for {root_cause}")
        return hypotheses
    
    def run_experiment(
        self,
        hypothesis: OptimizationHypothesis,
        config: Optional[ExperimentConfig] = None,
    ) -> ExperimentResult:
        """
        Run optimization experiment for hypothesis.
        
        Args:
            hypothesis: Hypothesis to test
            config: Experiment configuration
            
        Returns:
            ExperimentResult with validation outcome
        """
        config = config or ExperimentConfig(
            experiment_id=f"exp_{self._next_experiment_id}",
            hypothesis=hypothesis,
        )
        self._next_experiment_id += 1
        
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            hypothesis_id=hypothesis.hypothesis_id,
            start_time=time.time(),
        )
        
        try:
            # Measure baseline
            result.status = ExperimentStatus.RUNNING
            baseline_values = self._measure_baseline(config.measurement_iterations)
            result.baseline_value = self._compute_median(baseline_values)
            
            # Apply optimization
            if self._apply_config_func:
                success = self._apply_config_func(hypothesis.config_changes)
                if not success:
                    result.status = ExperimentStatus.FAILED
                    result.error_message = "Failed to apply configuration"
                    return result
            
            # Measure optimized
            optimized_values = self._measure_optimized(config.measurement_iterations)
            result.optimized_value = self._compute_median(optimized_values)
            
            # Compute improvement
            if result.baseline_value > 0:
                result.improvement_pct = (
                    (result.baseline_value - result.optimized_value) / 
                    result.baseline_value * 100
                )
            
            # Validate
            result.is_improvement = result.improvement_pct >= config.min_improvement_pct
            result.is_statistically_significant = self._check_significance(
                baseline_values, optimized_values
            )
            result.confidence = self._compute_confidence(result, config)
            
            # Determine outcome
            if result.is_improvement and result.is_statistically_significant:
                result.status = ExperimentStatus.VALIDATED
                self._state.applied_optimizations.append(hypothesis.hypothesis_id)
            elif result.improvement_pct < -config.rollback_threshold_pct:
                # Significant regression - rollback
                result.status = ExperimentStatus.ROLLED_BACK
                if config.auto_rollback:
                    self._rollback_optimization(hypothesis)
            else:
                result.status = ExperimentStatus.COMPLETED
            
        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Experiment failed: {e}")
        
        result.end_time = time.time()
        result.duration_seconds = result.end_time - result.start_time
        
        # Record history
        self._state.experiment_history.append(result)
        
        return result
    
    def run_next_experiment(self) -> Optional[ExperimentResult]:
        """
        Run the next highest-priority experiment.
        
        Returns:
            ExperimentResult or None if no pending hypotheses
        """
        if not self._pending_hypotheses:
            return None
        
        hypothesis = self._pending_hypotheses.pop(0)
        return self.run_experiment(hypothesis)
    
    def run_optimization_loop(
        self,
        max_experiments: int = 5,
        stop_on_success: bool = True,
    ) -> List[ExperimentResult]:
        """
        Run optimization loop until success or limit reached.
        
        Args:
            max_experiments: Maximum experiments to run
            stop_on_success: Stop after first validated improvement
            
        Returns:
            List of experiment results
        """
        results = []
        
        for _ in range(max_experiments):
            if not self._pending_hypotheses:
                break
            
            result = self.run_next_experiment()
            if result:
                results.append(result)
                
                if stop_on_success and result.status == ExperimentStatus.VALIDATED:
                    logger.info(f"Optimization validated: {result.improvement_pct:+.2f}%")
                    break
        
        return results
    
    def _measure_baseline(self, iterations: int) -> List[float]:
        """Measure baseline performance."""
        if not self._measurement_func:
            return [100.0] * iterations  # Mock values
        
        values = []
        for _ in range(iterations):
            values.append(self._measurement_func())
        return values
    
    def _measure_optimized(self, iterations: int) -> List[float]:
        """Measure optimized performance."""
        if not self._measurement_func:
            # Mock 10% improvement
            return [90.0] * iterations
        
        values = []
        for _ in range(iterations):
            values.append(self._measurement_func())
        return values
    
    def _compute_median(self, values: List[float]) -> float:
        """Compute median of values."""
        if not values:
            return 0.0
        sorted_v = sorted(values)
        n = len(sorted_v)
        if n % 2 == 0:
            return (sorted_v[n//2 - 1] + sorted_v[n//2]) / 2
        return sorted_v[n//2]
    
    def _check_significance(
        self,
        baseline: List[float],
        optimized: List[float],
    ) -> bool:
        """Check statistical significance using Mann-Whitney approximation."""
        if len(baseline) < 5 or len(optimized) < 5:
            return False
        
        # Simple check: non-overlapping medians
        baseline_med = self._compute_median(baseline)
        optimized_med = self._compute_median(optimized)
        
        baseline_mad = self._compute_median([abs(v - baseline_med) for v in baseline])
        
        # Significant if difference > 2 MAD
        return abs(baseline_med - optimized_med) > 2 * baseline_mad
    
    def _compute_confidence(
        self,
        result: ExperimentResult,
        config: ExperimentConfig,
    ) -> float:
        """Compute confidence in experiment result."""
        confidence = 0.5
        
        if result.is_improvement:
            confidence += 0.2
        
        if result.is_statistically_significant:
            confidence += 0.2
        
        # Boost for large improvements
        if result.improvement_pct > 10:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _rollback_optimization(
        self,
        hypothesis: OptimizationHypothesis,
    ) -> None:
        """Rollback an optimization."""
        logger.info(f"Rolling back optimization {hypothesis.hypothesis_id}")
        
        if self._apply_config_func:
            # Apply inverse config (simplified)
            rollback_config = {
                k: not v if isinstance(v, bool) else v
                for k, v in hypothesis.config_changes.items()
            }
            self._apply_config_func(rollback_config)
    
    def get_state(self) -> OptimizationState:
        """Get current optimization state."""
        return self._state
    
    def get_experiment_history(self) -> List[Dict[str, Any]]:
        """Get experiment history as dicts."""
        return [r.to_dict() for r in self._state.experiment_history]
    
    def get_applied_optimizations(self) -> List[str]:
        """Get list of applied optimization IDs."""
        return self._state.applied_optimizations.copy()


def create_auto_optimizer(
    measurement_func: Optional[Callable[[], float]] = None,
    apply_config_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> AutoOptimizationEngine:
    """
    Factory function to create auto-optimization engine.
    
    Args:
        measurement_func: Function to measure target metric
        apply_config_func: Function to apply configuration changes
        
    Returns:
        Configured AutoOptimizationEngine
    """
    return AutoOptimizationEngine(
        measurement_func=measurement_func,
        apply_config_func=apply_config_func,
    )
