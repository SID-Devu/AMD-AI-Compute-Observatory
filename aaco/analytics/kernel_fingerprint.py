"""
Kernel Family Fingerprinting (KFF).

Clusters and identifies kernel families by behavioral signatures:
- Name token analysis
- Duration distributions
- Grid/workgroup patterns
- Performance counter signatures
"""

import hashlib
import json
import logging
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ============================================================================
# Kernel Event
# ============================================================================

@dataclass
class KernelEvent:
    """A single kernel execution event."""
    name: str
    duration_us: float
    
    # Grid configuration
    grid_x: int = 1
    grid_y: int = 1
    grid_z: int = 1
    block_x: int = 1
    block_y: int = 1
    block_z: int = 1
    
    # Memory
    shared_memory_bytes: int = 0
    
    # Counters (if available)
    counters: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    stream_id: int = 0
    timestamp_ns: int = 0
    
    @property
    def total_threads(self) -> int:
        return (self.grid_x * self.grid_y * self.grid_z * 
                self.block_x * self.block_y * self.block_z)
    
    @property
    def total_blocks(self) -> int:
        return self.grid_x * self.grid_y * self.grid_z
    
    @property
    def threads_per_block(self) -> int:
        return self.block_x * self.block_y * self.block_z


# ============================================================================
# Kernel Family
# ============================================================================

class KernelCategory(str, Enum):
    """High-level kernel categories."""
    GEMM = "gemm"
    CONV = "conv"
    ATTENTION = "attention"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    MEMORY = "memory"
    NORMALIZATION = "norm"
    ACTIVATION = "activation"
    EMBEDDING = "embedding"
    SOFTMAX = "softmax"
    UNKNOWN = "unknown"


@dataclass
class KernelFingerprint:
    """
    Behavioral fingerprint for a kernel family.
    """
    # Identity
    family_id: str = ""
    representative_name: str = ""
    category: KernelCategory = KernelCategory.UNKNOWN
    
    # Name tokens (common tokens in kernel names)
    name_tokens: List[str] = field(default_factory=list)
    token_weights: Dict[str, float] = field(default_factory=dict)
    
    # Duration statistics
    duration_mean_us: float = 0.0
    duration_std_us: float = 0.0
    duration_p50_us: float = 0.0
    duration_p95_us: float = 0.0
    duration_p99_us: float = 0.0
    
    # Grid pattern statistics
    typical_grid_size: Tuple[int, int, int] = (1, 1, 1)
    typical_block_size: Tuple[int, int, int] = (1, 1, 1)
    threads_per_block_mean: float = 0.0
    total_threads_mean: float = 0.0
    
    # Performance signature
    counter_signature: Dict[str, float] = field(default_factory=dict)
    
    # Cluster info
    cluster_centroid: Optional[List[float]] = None
    member_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d
    
    def similarity_to(self, other: "KernelFingerprint") -> float:
        """Calculate similarity score to another fingerprint."""
        score = 0.0
        
        # Token overlap
        if self.name_tokens and other.name_tokens:
            common = set(self.name_tokens) & set(other.name_tokens)
            union = set(self.name_tokens) | set(other.name_tokens)
            score += 0.3 * (len(common) / max(len(union), 1))
        
        # Duration similarity (log-scale comparison)
        if self.duration_mean_us > 0 and other.duration_mean_us > 0:
            log_diff = abs(np.log10(self.duration_mean_us) - 
                          np.log10(other.duration_mean_us))
            score += 0.3 * max(0, 1 - log_diff / 3)  # 3 orders of magnitude = 0
        
        # Grid similarity
        if self.total_threads_mean > 0 and other.total_threads_mean > 0:
            log_diff = abs(np.log10(self.total_threads_mean) - 
                          np.log10(other.total_threads_mean))
            score += 0.2 * max(0, 1 - log_diff / 3)
        
        # Category match
        if self.category == other.category and self.category != KernelCategory.UNKNOWN:
            score += 0.2
        
        return score


@dataclass
class KernelFamilyRegistry:
    """Registry of known kernel families."""
    families: Dict[str, KernelFingerprint] = field(default_factory=dict)
    name_to_family: Dict[str, str] = field(default_factory=dict)  # kernel name -> family_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "families": {k: v.to_dict() for k, v in self.families.items()},
            "name_to_family": self.name_to_family,
        }
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "KernelFamilyRegistry":
        with open(path) as f:
            data = json.load(f)
        
        registry = cls()
        registry.name_to_family = data.get("name_to_family", {})
        
        for fid, fdata in data.get("families", {}).items():
            fp = KernelFingerprint()
            for k, v in fdata.items():
                if k == "category":
                    fp.category = KernelCategory(v)
                elif hasattr(fp, k):
                    setattr(fp, k, v)
            registry.families[fid] = fp
        
        return registry


# ============================================================================
# Name Token Analyzer
# ============================================================================

class NameTokenAnalyzer:
    """Extracts and analyzes tokens from kernel names."""
    
    # Common meaningful tokens and their categories
    TOKEN_CATEGORIES = {
        # GEMM family
        "gemm": KernelCategory.GEMM,
        "matmul": KernelCategory.GEMM,
        "sgemm": KernelCategory.GEMM,
        "hgemm": KernelCategory.GEMM,
        "dgemm": KernelCategory.GEMM,
        "bgemm": KernelCategory.GEMM,
        "batched_gemm": KernelCategory.GEMM,
        
        # Convolution family
        "conv": KernelCategory.CONV,
        "convolution": KernelCategory.CONV,
        "winograd": KernelCategory.CONV,
        "im2col": KernelCategory.CONV,
        "depthwise": KernelCategory.CONV,
        
        # Attention family
        "attention": KernelCategory.ATTENTION,
        "flash_attn": KernelCategory.ATTENTION,
        "self_attention": KernelCategory.ATTENTION,
        "multihead": KernelCategory.ATTENTION,
        "sdpa": KernelCategory.ATTENTION,
        
        # Elementwise
        "elementwise": KernelCategory.ELEMENTWISE,
        "binary": KernelCategory.ELEMENTWISE,
        "unary": KernelCategory.ELEMENTWISE,
        "add": KernelCategory.ELEMENTWISE,
        "mul": KernelCategory.ELEMENTWISE,
        "fused": KernelCategory.ELEMENTWISE,
        
        # Reduction
        "reduce": KernelCategory.REDUCTION,
        "sum": KernelCategory.REDUCTION,
        "mean": KernelCategory.REDUCTION,
        "max": KernelCategory.REDUCTION,
        "min": KernelCategory.REDUCTION,
        "argmax": KernelCategory.REDUCTION,
        
        # Memory
        "copy": KernelCategory.MEMORY,
        "transpose": KernelCategory.MEMORY,
        "permute": KernelCategory.MEMORY,
        "gather": KernelCategory.MEMORY,
        "scatter": KernelCategory.MEMORY,
        "index": KernelCategory.MEMORY,
        
        # Normalization
        "layernorm": KernelCategory.NORMALIZATION,
        "batchnorm": KernelCategory.NORMALIZATION,
        "groupnorm": KernelCategory.NORMALIZATION,
        "rmsnorm": KernelCategory.NORMALIZATION,
        "normalize": KernelCategory.NORMALIZATION,
        
        # Activation
        "relu": KernelCategory.ACTIVATION,
        "gelu": KernelCategory.ACTIVATION,
        "silu": KernelCategory.ACTIVATION,
        "sigmoid": KernelCategory.ACTIVATION,
        "tanh": KernelCategory.ACTIVATION,
        "swish": KernelCategory.ACTIVATION,
        
        # Embedding
        "embedding": KernelCategory.EMBEDDING,
        "lookup": KernelCategory.EMBEDDING,
        
        # Softmax
        "softmax": KernelCategory.SOFTMAX,
        "log_softmax": KernelCategory.SOFTMAX,
    }
    
    def __init__(self):
        self._token_pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9]*')
    
    def extract_tokens(self, name: str) -> List[str]:
        """Extract meaningful tokens from kernel name."""
        # Convert camelCase and PascalCase to tokens
        # Split on underscores, numbers, special chars
        
        # First expand camelCase
        expanded = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        
        # Extract all letter sequences
        tokens = self._token_pattern.findall(expanded.lower())
        
        # Filter very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def categorize_kernel(self, name: str) -> KernelCategory:
        """Determine kernel category from name."""
        tokens = self.extract_tokens(name)
        name_lower = name.lower()
        
        # Check for exact matches first
        for pattern, category in self.TOKEN_CATEGORIES.items():
            if pattern in name_lower:
                return category
        
        # Check tokens
        for token in tokens:
            if token in self.TOKEN_CATEGORIES:
                return self.TOKEN_CATEGORIES[token]
        
        return KernelCategory.UNKNOWN
    
    def compute_token_weights(self, names: List[str]) -> Dict[str, float]:
        """Compute token weights based on frequency (TF-IDF style)."""
        all_tokens = []
        name_token_sets = []
        
        for name in names:
            tokens = self.extract_tokens(name)
            all_tokens.extend(tokens)
            name_token_sets.append(set(tokens))
        
        # Term frequency
        token_freq = defaultdict(int)
        for t in all_tokens:
            token_freq[t] += 1
        
        # Document frequency (how many names contain this token)
        doc_freq = defaultdict(int)
        for token_set in name_token_sets:
            for t in token_set:
                doc_freq[t] += 1
        
        # TF-IDF weights
        n_docs = len(names)
        weights = {}
        for token, tf in token_freq.items():
            df = doc_freq[token]
            idf = np.log(n_docs / (df + 1)) + 1
            weights[token] = tf * idf
        
        # Normalize
        max_weight = max(weights.values()) if weights else 1
        weights = {t: w / max_weight for t, w in weights.items()}
        
        return weights


# ============================================================================
# Kernel Family Fingerprinter
# ============================================================================

class KernelFamilyFingerprinter:
    """
    Builds kernel family fingerprints from execution traces.
    
    Clusters kernels by behavioral similarity and creates fingerprints
    that can be used to:
    - Identify kernel types at runtime
    - Track performance drift over time
    - Group similar kernels for analysis
    
    Usage:
        fingerprinter = KernelFamilyFingerprinter()
        fingerprinter.add_events(kernel_events)
        registry = fingerprinter.build_registry()
        registry.save(Path("kernel_families.json"))
    """
    
    def __init__(self, 
                 min_samples: int = 5,
                 cluster_threshold: float = 0.5):
        """
        Args:
            min_samples: Minimum samples to create a family
            cluster_threshold: Threshold for hierarchical clustering
        """
        self.min_samples = min_samples
        self.cluster_threshold = cluster_threshold
        
        self._events_by_name: Dict[str, List[KernelEvent]] = defaultdict(list)
        self._name_analyzer = NameTokenAnalyzer()
    
    def add_event(self, event: KernelEvent) -> None:
        """Add a single kernel event."""
        self._events_by_name[event.name].append(event)
    
    def add_events(self, events: List[KernelEvent]) -> None:
        """Add multiple kernel events."""
        for event in events:
            self.add_event(event)
    
    def build_registry(self) -> KernelFamilyRegistry:
        """Build kernel family registry from collected events."""
        registry = KernelFamilyRegistry()
        
        # First, create fingerprints for each unique kernel name
        name_fingerprints: Dict[str, KernelFingerprint] = {}
        
        for name, events in self._events_by_name.items():
            if len(events) < self.min_samples:
                continue
            
            fp = self._create_fingerprint_from_events(name, events)
            name_fingerprints[name] = fp
        
        # Cluster similar kernels into families
        if len(name_fingerprints) > 1:
            families = self._cluster_fingerprints(name_fingerprints)
        else:
            # Each kernel is its own family
            families = {name: [name] for name in name_fingerprints}
        
        # Build final registry
        for family_id, member_names in families.items():
            # Merge fingerprints for family
            family_fp = self._merge_fingerprints(
                family_id, 
                [name_fingerprints[n] for n in member_names]
            )
            registry.families[family_id] = family_fp
            
            # Map each member name to family
            for name in member_names:
                registry.name_to_family[name] = family_id
        
        logger.info(f"Built registry with {len(registry.families)} families "
                   f"from {len(name_fingerprints)} unique kernels")
        
        return registry
    
    def _create_fingerprint_from_events(self, 
                                        name: str, 
                                        events: List[KernelEvent]) -> KernelFingerprint:
        """Create fingerprint from kernel events."""
        fp = KernelFingerprint()
        fp.representative_name = name
        
        # Name analysis
        fp.name_tokens = self._name_analyzer.extract_tokens(name)
        fp.category = self._name_analyzer.categorize_kernel(name)
        
        # Duration statistics
        durations = [e.duration_us for e in events]
        fp.duration_mean_us = statistics.mean(durations)
        fp.duration_std_us = statistics.stdev(durations) if len(durations) > 1 else 0
        fp.duration_p50_us = statistics.median(durations)
        
        sorted_durations = sorted(durations)
        idx_95 = int(0.95 * len(sorted_durations))
        idx_99 = int(0.99 * len(sorted_durations))
        fp.duration_p95_us = sorted_durations[min(idx_95, len(sorted_durations) - 1)]
        fp.duration_p99_us = sorted_durations[min(idx_99, len(sorted_durations) - 1)]
        
        # Grid statistics
        threads_per_block = [e.threads_per_block for e in events]
        total_threads = [e.total_threads for e in events]
        
        fp.threads_per_block_mean = statistics.mean(threads_per_block)
        fp.total_threads_mean = statistics.mean(total_threads)
        
        # Most common grid/block sizes
        grid_counts = defaultdict(int)
        block_counts = defaultdict(int)
        for e in events:
            grid_counts[(e.grid_x, e.grid_y, e.grid_z)] += 1
            block_counts[(e.block_x, e.block_y, e.block_z)] += 1
        
        fp.typical_grid_size = max(grid_counts, key=grid_counts.get)
        fp.typical_block_size = max(block_counts, key=block_counts.get)
        
        # Counter signature (average counters)
        counter_sums = defaultdict(float)
        counter_counts = defaultdict(int)
        for e in events:
            for k, v in e.counters.items():
                counter_sums[k] += v
                counter_counts[k] += 1
        
        fp.counter_signature = {
            k: counter_sums[k] / counter_counts[k] 
            for k in counter_sums
        }
        
        fp.member_count = len(events)
        
        # Create feature vector for clustering
        fp.cluster_centroid = self._fingerprint_to_vector(fp)
        
        return fp
    
    def _fingerprint_to_vector(self, fp: KernelFingerprint) -> List[float]:
        """Convert fingerprint to feature vector for clustering."""
        return [
            np.log10(fp.duration_mean_us + 1),
            np.log10(fp.duration_std_us + 1),
            np.log10(fp.total_threads_mean + 1),
            np.log10(fp.threads_per_block_mean + 1),
            float(hash(fp.category.value) % 100) / 100,  # Category as numeric
        ]
    
    def _cluster_fingerprints(self, 
                              name_fingerprints: Dict[str, KernelFingerprint]
                              ) -> Dict[str, List[str]]:
        """Cluster fingerprints into families using hierarchical clustering."""
        names = list(name_fingerprints.keys())
        vectors = np.array([fp.cluster_centroid for fp in name_fingerprints.values()])
        
        if len(vectors) < 2:
            return {names[0]: names}
        
        try:
            # Hierarchical clustering
            clusters = fclusterdata(
                vectors, 
                t=self.cluster_threshold, 
                criterion='distance',
                method='average',
                metric='euclidean'
            )
            
            # Group names by cluster
            families: Dict[str, List[str]] = defaultdict(list)
            for name, cluster_id in zip(names, clusters):
                # Use first member name as family ID
                family_id = f"family_{cluster_id}"
                families[family_id].append(name)
            
            # Rename families with representative name
            renamed_families = {}
            for fid, members in families.items():
                # Use the most common/representative kernel name
                representative = min(members, key=len)
                tokens = self._name_analyzer.extract_tokens(representative)
                if tokens:
                    family_name = "_".join(tokens[:3])
                else:
                    family_name = representative[:20]
                
                family_hash = hashlib.md5(family_name.encode()).hexdigest()[:8]
                new_fid = f"{family_name}_{family_hash}"
                renamed_families[new_fid] = members
            
            return renamed_families
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            # Fallback: each kernel is its own family
            return {n: [n] for n in names}
    
    def _merge_fingerprints(self, 
                            family_id: str,
                            fingerprints: List[KernelFingerprint]) -> KernelFingerprint:
        """Merge multiple fingerprints into one family fingerprint."""
        merged = KernelFingerprint()
        merged.family_id = family_id
        
        # Take first representative
        merged.representative_name = fingerprints[0].representative_name
        
        # Merge tokens (union)
        all_tokens = set()
        for fp in fingerprints:
            all_tokens.update(fp.name_tokens)
        merged.name_tokens = list(all_tokens)
        
        # Category (most common)
        categories = [fp.category for fp in fingerprints]
        merged.category = max(set(categories), key=categories.count)
        
        # Duration stats (weighted average by member count)
        total_members = sum(fp.member_count for fp in fingerprints)
        merged.duration_mean_us = sum(
            fp.duration_mean_us * fp.member_count / total_members 
            for fp in fingerprints
        )
        merged.duration_p50_us = statistics.median(
            [fp.duration_p50_us for fp in fingerprints]
        )
        merged.duration_p95_us = max(fp.duration_p95_us for fp in fingerprints)
        merged.duration_p99_us = max(fp.duration_p99_us for fp in fingerprints)
        
        # Grid stats
        merged.threads_per_block_mean = statistics.mean(
            [fp.threads_per_block_mean for fp in fingerprints]
        )
        merged.total_threads_mean = statistics.mean(
            [fp.total_threads_mean for fp in fingerprints]
        )
        
        merged.member_count = total_members
        
        return merged


# ============================================================================
# Drift Detection
# ============================================================================

def detect_fingerprint_drift(baseline: KernelFingerprint,
                             current: KernelFingerprint,
                             threshold: float = 0.2) -> Dict[str, Any]:
    """
    Detect drift between baseline and current fingerprints.
    
    Returns:
        Dictionary with drift analysis
    """
    drift = {
        "has_drift": False,
        "duration_drift_pct": 0.0,
        "thread_drift_pct": 0.0,
        "details": [],
    }
    
    # Duration drift
    if baseline.duration_mean_us > 0:
        dur_pct = (current.duration_mean_us - baseline.duration_mean_us) / baseline.duration_mean_us
        drift["duration_drift_pct"] = dur_pct * 100
        
        if abs(dur_pct) > threshold:
            drift["has_drift"] = True
            drift["details"].append(
                f"Duration changed {dur_pct*100:+.1f}% "
                f"({baseline.duration_mean_us:.1f}us -> {current.duration_mean_us:.1f}us)"
            )
    
    # Thread count drift
    if baseline.total_threads_mean > 0:
        thread_pct = (current.total_threads_mean - baseline.total_threads_mean) / baseline.total_threads_mean
        drift["thread_drift_pct"] = thread_pct * 100
        
        if abs(thread_pct) > threshold:
            drift["has_drift"] = True
            drift["details"].append(
                f"Thread count changed {thread_pct*100:+.1f}% "
                f"({baseline.total_threads_mean:.0f} -> {current.total_threads_mean:.0f})"
            )
    
    return drift
