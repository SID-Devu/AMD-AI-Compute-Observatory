"""
AACO-SIGMA Evidence Collector

Collects and chains evidence supporting root cause diagnoses.
Provides traceable evidence trails for debugging.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto
import time
import hashlib


class EvidenceType(Enum):
    """Types of evidence."""
    COUNTER_VALUE = auto()      # Hardware counter reading
    TIMING_MEASUREMENT = auto() # Timing data
    PROFILE_DATA = auto()       # Profiler output
    TRACE_EVENT = auto()        # Trace event
    CONFIGURATION = auto()      # Configuration setting
    COMPARISON = auto()         # Comparison to baseline/expected
    LOG_ENTRY = auto()          # Log message
    METRIC = auto()             # Computed metric
    EXTERNAL = auto()           # External data source


class EvidenceStrength(Enum):
    """Strength of evidence."""
    STRONG = auto()      # Direct, definitive evidence
    MODERATE = auto()    # Supporting evidence
    WEAK = auto()        # Circumstantial evidence
    INFERRED = auto()    # Derived/inferred evidence


@dataclass
class Evidence:
    """A piece of evidence supporting a root cause."""
    
    # Identity
    evidence_id: str
    evidence_type: EvidenceType
    
    # Content
    description: str
    value: Any = None
    unit: str = ""
    
    # Source
    source: str = ""           # Where evidence came from
    source_timestamp: float = 0.0
    
    # Strength
    strength: EvidenceStrength = EvidenceStrength.MODERATE
    
    # Related
    related_evidence: List[str] = field(default_factory=list)
    supports_cause: str = ""   # Cause ID this supports
    
    # Thresholds for interpretation
    threshold_exceeded: bool = False
    threshold_value: float = 0.0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.evidence_id,
            "type": self.evidence_type.name,
            "description": self.description,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "strength": self.strength.name,
        }


@dataclass
class EvidenceChain:
    """A chain of evidence supporting a diagnosis."""
    
    chain_id: str
    cause_id: str
    
    # Ordered evidence
    evidence_list: List[Evidence] = field(default_factory=list)
    
    # Chain strength (aggregate)
    chain_strength: float = 0.0
    
    # Narrative
    narrative: str = ""
    
    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence to the chain."""
        evidence.related_evidence.append(self.chain_id)
        self.evidence_list.append(evidence)
        self._update_strength()
    
    def _update_strength(self) -> None:
        """Update chain strength based on evidence."""
        if not self.evidence_list:
            self.chain_strength = 0.0
            return
        
        # Weight by evidence strength
        weights = {
            EvidenceStrength.STRONG: 1.0,
            EvidenceStrength.MODERATE: 0.7,
            EvidenceStrength.WEAK: 0.4,
            EvidenceStrength.INFERRED: 0.2,
        }
        
        total_weight = sum(
            weights.get(e.strength, 0.5) for e in self.evidence_list
        )
        
        # Normalize to 0-1, capped at 1
        self.chain_strength = min(1.0, total_weight / len(self.evidence_list))


class EvidenceCollector:
    """
    Collects evidence from multiple sources to support root cause analysis.
    
    Evidence sources:
    - Hardware counters
    - Timing measurements
    - Profiler data
    - Trace events
    - Configuration
    - Baselines/comparisons
    """
    
    def __init__(self):
        self.evidence_store: Dict[str, Evidence] = {}
        self.chains: Dict[str, EvidenceChain] = {}
        self._evidence_counter = 0
    
    def _generate_id(self, prefix: str = "ev") -> str:
        """Generate unique evidence ID."""
        self._evidence_counter += 1
        return f"{prefix}_{self._evidence_counter:06d}"
    
    def collect_counter(self,
                        counter_name: str,
                        value: float,
                        threshold: float = 0.0,
                        kernel_name: str = "") -> Evidence:
        """Collect hardware counter evidence."""
        evidence = Evidence(
            evidence_id=self._generate_id("counter"),
            evidence_type=EvidenceType.COUNTER_VALUE,
            description=f"{counter_name} = {value:.2f}" + 
                        (f" (threshold: {threshold})" if threshold else ""),
            value=value,
            source=f"hardware_counter:{kernel_name}" if kernel_name else "hardware_counter",
            source_timestamp=time.time(),
            strength=EvidenceStrength.STRONG,
            threshold_exceeded=value > threshold if threshold else False,
            threshold_value=threshold,
            tags=["counter", counter_name],
        )
        
        self.evidence_store[evidence.evidence_id] = evidence
        return evidence
    
    def collect_timing(self,
                       metric_name: str,
                       value_ms: float,
                       baseline_ms: float = 0.0,
                       kernel_name: str = "") -> Evidence:
        """Collect timing measurement evidence."""
        description = f"{metric_name}: {value_ms:.3f}ms"
        if baseline_ms > 0:
            delta_pct = ((value_ms - baseline_ms) / baseline_ms) * 100
            description += f" (baseline: {baseline_ms:.3f}ms, delta: {delta_pct:+.1f}%)"
        
        evidence = Evidence(
            evidence_id=self._generate_id("timing"),
            evidence_type=EvidenceType.TIMING_MEASUREMENT,
            description=description,
            value=value_ms,
            unit="ms",
            source=f"timing:{kernel_name}" if kernel_name else "timing",
            source_timestamp=time.time(),
            strength=EvidenceStrength.STRONG,
            tags=["timing", metric_name],
        )
        
        if baseline_ms > 0:
            evidence.threshold_value = baseline_ms
            evidence.threshold_exceeded = value_ms > baseline_ms * 1.1  # 10% threshold
        
        self.evidence_store[evidence.evidence_id] = evidence
        return evidence
    
    def collect_metric(self,
                       name: str,
                       value: float,
                       expected: float = 0.0,
                       unit: str = "") -> Evidence:
        """Collect computed metric evidence."""
        description = f"{name}: {value:.2f}{unit}"
        if expected > 0:
            ratio = value / expected
            description += f" (expected: {expected:.2f}, ratio: {ratio:.2f})"
        
        evidence = Evidence(
            evidence_id=self._generate_id("metric"),
            evidence_type=EvidenceType.METRIC,
            description=description,
            value=value,
            unit=unit,
            source="computed_metric",
            source_timestamp=time.time(),
            strength=EvidenceStrength.MODERATE,
            tags=["metric", name],
        )
        
        if expected > 0:
            evidence.threshold_value = expected
            evidence.threshold_exceeded = value > expected * 1.2  # 20% threshold
        
        self.evidence_store[evidence.evidence_id] = evidence
        return evidence
    
    def collect_comparison(self,
                           metric_name: str,
                           current: float,
                           baseline: float,
                           source: str = "baseline") -> Evidence:
        """Collect comparison evidence."""
        if baseline != 0:
            delta_pct = ((current - baseline) / baseline) * 100
        else:
            delta_pct = 0
        
        direction = "increased" if delta_pct > 0 else "decreased"
        
        evidence = Evidence(
            evidence_id=self._generate_id("compare"),
            evidence_type=EvidenceType.COMPARISON,
            description=f"{metric_name} {direction} by {abs(delta_pct):.1f}% "
                        f"({baseline:.3f} -> {current:.3f})",
            value=delta_pct,
            unit="%",
            source=source,
            source_timestamp=time.time(),
            strength=EvidenceStrength.STRONG if abs(delta_pct) > 10 else EvidenceStrength.MODERATE,
            threshold_exceeded=abs(delta_pct) > 10,
            tags=["comparison", metric_name],
        )
        
        self.evidence_store[evidence.evidence_id] = evidence
        return evidence
    
    def collect_trace_event(self,
                            event_name: str,
                            duration_us: float,
                            track: str = "",
                            metadata: Optional[Dict[str, Any]] = None) -> Evidence:
        """Collect trace event evidence."""
        evidence = Evidence(
            evidence_id=self._generate_id("trace"),
            evidence_type=EvidenceType.TRACE_EVENT,
            description=f"Trace event '{event_name}' duration: {duration_us:.1f}µs",
            value=duration_us,
            unit="µs",
            source=f"trace:{track}" if track else "trace",
            source_timestamp=time.time(),
            strength=EvidenceStrength.STRONG,
            tags=["trace", event_name],
        )
        
        self.evidence_store[evidence.evidence_id] = evidence
        return evidence
    
    def create_chain(self, cause_id: str, narrative: str = "") -> EvidenceChain:
        """Create an evidence chain for a cause."""
        chain = EvidenceChain(
            chain_id=self._generate_id("chain"),
            cause_id=cause_id,
            narrative=narrative,
        )
        
        self.chains[chain.chain_id] = chain
        return chain
    
    def get_evidence_for_cause(self, cause_id: str) -> List[Evidence]:
        """Get all evidence supporting a cause."""
        return [
            e for e in self.evidence_store.values()
            if e.supports_cause == cause_id
        ]
    
    def get_chain_for_cause(self, cause_id: str) -> Optional[EvidenceChain]:
        """Get evidence chain for a cause."""
        for chain in self.chains.values():
            if chain.cause_id == cause_id:
                return chain
        return None
    
    def summarize_evidence(self) -> Dict[str, Any]:
        """Generate evidence summary."""
        type_counts: Dict[str, int] = {}
        strength_counts: Dict[str, int] = {}
        
        for evidence in self.evidence_store.values():
            type_name = evidence.evidence_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            strength_name = evidence.strength.name
            strength_counts[strength_name] = strength_counts.get(strength_name, 0) + 1
        
        return {
            "total_evidence": len(self.evidence_store),
            "total_chains": len(self.chains),
            "by_type": type_counts,
            "by_strength": strength_counts,
            "threshold_exceeded": sum(
                1 for e in self.evidence_store.values() 
                if e.threshold_exceeded
            ),
        }
    
    def export_chain(self, chain: EvidenceChain) -> Dict[str, Any]:
        """Export evidence chain for reporting."""
        return {
            "chain_id": chain.chain_id,
            "cause_id": chain.cause_id,
            "strength": chain.chain_strength,
            "narrative": chain.narrative,
            "evidence": [e.to_dict() for e in chain.evidence_list],
        }
