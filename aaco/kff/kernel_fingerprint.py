"""
AACO-SIGMA Kernel Fingerprint Framework

Generates unique fingerprints for GPU kernels based on:
- Kernel name and mangled signature
- Launch configuration (grid, block)
- Argument patterns
- GPU architecture
- Timing characteristics
"""

import hashlib
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Set
from pathlib import Path
from collections import defaultdict
from enum import Enum, auto


class FingerprintVersion(Enum):
    """Fingerprint algorithm versions."""
    V1 = 1  # Basic name + config hash
    V2 = 2  # + timing characteristics
    V3 = 3  # + counter signature


@dataclass
class KernelSignature:
    """
    Raw kernel signature from profiler data.
    
    Contains everything we know about a kernel invocation.
    """
    # Identity
    name: str
    mangled_name: str = ""
    module_name: str = ""
    
    # Launch configuration
    grid_x: int = 0
    grid_y: int = 0
    grid_z: int = 0
    block_x: int = 0
    block_y: int = 0
    block_z: int = 0
    
    shared_memory_bytes: int = 0
    registers_per_thread: int = 0
    
    # Runtime info
    stream_id: int = 0
    queue_id: int = 0
    
    # Arguments (hashed for privacy)
    arg_count: int = 0
    arg_sizes: List[int] = field(default_factory=list)
    
    # Timing (for V2+ fingerprints)
    duration_ns: int = 0
    
    @property
    def grid_size(self) -> int:
        """Total grid size."""
        return max(1, self.grid_x) * max(1, self.grid_y) * max(1, self.grid_z)
    
    @property
    def block_size(self) -> int:
        """Total block size."""
        return max(1, self.block_x) * max(1, self.block_y) * max(1, self.block_z)
    
    @property
    def total_threads(self) -> int:
        """Total threads launched."""
        return self.grid_size * self.block_size


@dataclass
class KernelFingerprint:
    """
    Stable fingerprint for kernel identification.
    
    Can be used to match kernels across:
    - Different runs of the same model
    - Different input sizes (when config is similar)
    - Different hardware (same kernel, different performance)
    """
    # Fingerprint data
    fingerprint_id: str  # Primary hash
    short_id: str        # 8-char short hash
    version: FingerprintVersion = FingerprintVersion.V3
    
    # Source signature
    kernel_name: str = ""
    canonical_name: str = ""  # Demangled/normalized
    
    # Configuration hash (grid/block independent of actual values)
    config_class: str = ""    # e.g., "3D_grid_1D_block"
    
    # Family hint
    family_id: str = ""       # e.g., "gemm", "conv2d", "attention"
    
    # Timing class (V2+)
    timing_class: str = ""    # e.g., "micro", "small", "medium", "large"
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["version"] = self.version.value
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'KernelFingerprint':
        d = d.copy()
        d["version"] = FingerprintVersion(d.get("version", 3))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FingerprintGenerator:
    """
    Generates fingerprints from kernel signatures.
    
    Uses multi-level hashing to create fingerprints that are:
    - Stable across runs with same config
    - Distinguishing between different kernels
    - Family-aware for related kernels
    """
    
    # Kernel name patterns for family detection
    FAMILY_PATTERNS = {
        "gemm": [
            r"gemm", r"matmul", r"_mm_", r"sgemm", r"dgemm", r"hgemm",
            r"rocblas.*gemm", r"ck::.*Gemm"
        ],
        "conv": [
            r"conv", r"convolution", r"winograd", r"im2col",
            r"miopen.*conv", r"ck::.*Conv"
        ],
        "attention": [
            r"attention", r"softmax", r"self_attn", r"cross_attn",
            r"flash.*attn", r"scaled_dot"
        ],
        "normalization": [
            r"layernorm", r"batchnorm", r"groupnorm", r"rmsnorm",
            r"layer_norm", r"batch_norm"
        ],
        "elementwise": [
            r"relu", r"gelu", r"silu", r"sigmoid", r"tanh",
            r"add", r"mul", r"fused", r"elementwise"
        ],
        "reduction": [
            r"reduce", r"sum", r"mean", r"max", r"min",
            r"argmax", r"argmin"
        ],
        "embedding": [
            r"embed", r"lookup", r"gather", r"index_select"
        ],
        "memory": [
            r"copy", r"transpose", r"permute", r"reshape",
            r"contiguous", r"view"
        ],
    }
    
    # Timing thresholds (nanoseconds)
    TIMING_CLASSES = {
        "micro": (0, 1_000),           # < 1 µs
        "small": (1_000, 10_000),       # 1-10 µs
        "medium": (10_000, 100_000),    # 10-100 µs
        "large": (100_000, 1_000_000),  # 100 µs - 1 ms
        "xlarge": (1_000_000, float('inf')),  # > 1 ms
    }
    
    def __init__(self, version: FingerprintVersion = FingerprintVersion.V3):
        self.version = version
        self._family_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for family detection."""
        compiled = {}
        for family, patterns in self.FAMILY_PATTERNS.items():
            compiled[family] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    def generate(self, signature: KernelSignature) -> KernelFingerprint:
        """Generate fingerprint from kernel signature."""
        # Compute canonical name
        canonical = self._canonicalize_name(signature.name)
        
        # Detect family
        family_id = self._detect_family(signature.name)
        
        # Compute configuration class
        config_class = self._compute_config_class(signature)
        
        # Compute timing class
        timing_class = self._compute_timing_class(signature.duration_ns)
        
        # Generate hash components
        hash_input = self._build_hash_input(
            canonical, config_class, family_id, timing_class
        )
        
        # Generate fingerprint ID
        fingerprint_id = hashlib.sha256(hash_input.encode()).hexdigest()
        short_id = fingerprint_id[:8]
        
        return KernelFingerprint(
            fingerprint_id=fingerprint_id,
            short_id=short_id,
            version=self.version,
            kernel_name=signature.name,
            canonical_name=canonical,
            config_class=config_class,
            family_id=family_id,
            timing_class=timing_class,
            metadata={
                "grid_size": signature.grid_size,
                "block_size": signature.block_size,
                "shared_memory": signature.shared_memory_bytes,
            }
        )
    
    def _canonicalize_name(self, name: str) -> str:
        """Convert kernel name to canonical form."""
        # Remove common prefixes
        canonical = name
        prefixes = ["__", "_Z", "void ", "kernel_", "hip_"]
        for prefix in prefixes:
            if canonical.startswith(prefix):
                canonical = canonical[len(prefix):]
        
        # Remove template arguments for shorter names
        canonical = re.sub(r'<[^>]+>', '<T>', canonical)
        
        # Remove parameter types
        canonical = re.sub(r'\([^)]*\)', '()', canonical)
        
        return canonical
    
    def _detect_family(self, name: str) -> str:
        """Detect kernel family from name."""
        name_lower = name.lower()
        
        for family, patterns in self._family_patterns.items():
            for pattern in patterns:
                if pattern.search(name_lower):
                    return family
        
        return "unknown"
    
    def _compute_config_class(self, sig: KernelSignature) -> str:
        """Compute launch configuration class."""
        # Grid dimensionality
        grid_dims = sum([
            sig.grid_x > 1,
            sig.grid_y > 1,
            sig.grid_z > 1
        ])
        
        # Block dimensionality
        block_dims = sum([
            sig.block_x > 1,
            sig.block_y > 1,
            sig.block_z > 1
        ])
        
        # Size class
        total = sig.total_threads
        if total < 256:
            size = "tiny"
        elif total < 4096:
            size = "small"
        elif total < 65536:
            size = "medium"
        elif total < 1048576:
            size = "large"
        else:
            size = "xlarge"
        
        return f"{grid_dims}D_grid_{block_dims}D_block_{size}"
    
    def _compute_timing_class(self, duration_ns: int) -> str:
        """Compute timing class from duration."""
        for class_name, (min_ns, max_ns) in self.TIMING_CLASSES.items():
            if min_ns <= duration_ns < max_ns:
                return class_name
        return "unknown"
    
    def _build_hash_input(self, canonical: str, config_class: str,
                         family_id: str, timing_class: str) -> str:
        """Build hash input string."""
        components = [canonical, config_class, family_id]
        
        if self.version.value >= 2:
            components.append(timing_class)
        
        return "|".join(components)


class FingerprintMatcher:
    """
    Matches kernels to fingerprints.
    
    Supports fuzzy matching for kernels that are similar but not identical.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self._generator = FingerprintGenerator()
    
    def exact_match(self, fp1: KernelFingerprint, fp2: KernelFingerprint) -> bool:
        """Check for exact fingerprint match."""
        return fp1.fingerprint_id == fp2.fingerprint_id
    
    def fuzzy_match(self, fp1: KernelFingerprint, fp2: KernelFingerprint) -> float:
        """Compute fuzzy match score (0-1)."""
        score = 0.0
        weights = {
            "canonical": 0.4,
            "family": 0.25,
            "config": 0.25,
            "timing": 0.1,
        }
        
        # Canonical name similarity
        if fp1.canonical_name == fp2.canonical_name:
            score += weights["canonical"]
        elif self._string_similarity(fp1.canonical_name, fp2.canonical_name) > 0.7:
            score += weights["canonical"] * 0.5
        
        # Family match
        if fp1.family_id == fp2.family_id:
            score += weights["family"]
        
        # Config class match
        if fp1.config_class == fp2.config_class:
            score += weights["config"]
        elif self._config_compatible(fp1.config_class, fp2.config_class):
            score += weights["config"] * 0.5
        
        # Timing class match
        if fp1.timing_class == fp2.timing_class:
            score += weights["timing"]
        elif self._timing_adjacent(fp1.timing_class, fp2.timing_class):
            score += weights["timing"] * 0.5
        
        return score
    
    def is_match(self, fp1: KernelFingerprint, fp2: KernelFingerprint) -> bool:
        """Check if two fingerprints match (above threshold)."""
        if self.exact_match(fp1, fp2):
            return True
        return self.fuzzy_match(fp1, fp2) >= self.threshold
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity using longest common subsequence."""
        if not s1 or not s2:
            return 0.0
        
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return 2.0 * lcs_length / (m + n)
    
    def _config_compatible(self, c1: str, c2: str) -> bool:
        """Check if configurations are compatible."""
        # Same dimensionality but different size
        parts1 = c1.split("_")
        parts2 = c2.split("_")
        
        if len(parts1) >= 4 and len(parts2) >= 4:
            return parts1[:3] == parts2[:3]  # Same grid/block dims
        return False
    
    def _timing_adjacent(self, t1: str, t2: str) -> bool:
        """Check if timing classes are adjacent."""
        classes = ["micro", "small", "medium", "large", "xlarge"]
        try:
            idx1 = classes.index(t1)
            idx2 = classes.index(t2)
            return abs(idx1 - idx2) <= 1
        except ValueError:
            return False


class FingerprintDatabase:
    """
    Database of known kernel fingerprints.
    
    Supports:
    - Fingerprint storage and retrieval
    - Family-based lookup
    - Fuzzy matching
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        self._fingerprints: Dict[str, KernelFingerprint] = {}
        self._by_family: Dict[str, Set[str]] = defaultdict(set)
        self._by_canonical: Dict[str, Set[str]] = defaultdict(set)
        
        self._matcher = FingerprintMatcher()
        
        if db_path and db_path.exists():
            self.load(db_path)
    
    def add(self, fp: KernelFingerprint) -> None:
        """Add fingerprint to database."""
        self._fingerprints[fp.fingerprint_id] = fp
        self._by_family[fp.family_id].add(fp.fingerprint_id)
        self._by_canonical[fp.canonical_name].add(fp.fingerprint_id)
    
    def get(self, fingerprint_id: str) -> Optional[KernelFingerprint]:
        """Get fingerprint by ID."""
        return self._fingerprints.get(fingerprint_id)
    
    def get_by_short_id(self, short_id: str) -> Optional[KernelFingerprint]:
        """Get fingerprint by short ID."""
        for fp in self._fingerprints.values():
            if fp.short_id == short_id:
                return fp
        return None
    
    def find_by_family(self, family_id: str) -> List[KernelFingerprint]:
        """Find all fingerprints in a family."""
        fp_ids = self._by_family.get(family_id, set())
        return [self._fingerprints[fid] for fid in fp_ids]
    
    def find_match(self, fp: KernelFingerprint) -> Optional[KernelFingerprint]:
        """Find matching fingerprint (exact or fuzzy)."""
        # Try exact match first
        exact = self.get(fp.fingerprint_id)
        if exact:
            return exact
        
        # Try canonical name match
        candidates = self._by_canonical.get(fp.canonical_name, set())
        for cand_id in candidates:
            cand = self._fingerprints[cand_id]
            if self._matcher.is_match(fp, cand):
                return cand
        
        # Fuzzy match in same family
        family_cands = self._by_family.get(fp.family_id, set())
        best_match = None
        best_score = 0.0
        
        for cand_id in family_cands:
            cand = self._fingerprints[cand_id]
            score = self._matcher.fuzzy_match(fp, cand)
            if score > best_score and score >= self._matcher.threshold:
                best_score = score
                best_match = cand
        
        return best_match
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save database to file."""
        path = path or self.db_path
        if not path:
            return
        
        data = {
            "version": "1.0",
            "fingerprints": [fp.to_dict() for fp in self._fingerprints.values()]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path) -> None:
        """Load database from file."""
        with open(path) as f:
            data = json.load(f)
        
        for fp_dict in data.get("fingerprints", []):
            fp = KernelFingerprint.from_dict(fp_dict)
            self.add(fp)
    
    @property
    def size(self) -> int:
        """Number of fingerprints in database."""
        return len(self._fingerprints)
    
    def get_families(self) -> List[str]:
        """Get all family IDs."""
        return list(self._by_family.keys())
    
    def get_family_stats(self) -> Dict[str, int]:
        """Get kernel count per family."""
        return {family: len(fps) for family, fps in self._by_family.items()}
