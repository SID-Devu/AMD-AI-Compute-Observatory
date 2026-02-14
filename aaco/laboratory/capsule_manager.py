# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Capsule Manager

Manages execution capsules, baselines, and environment comparisons.
"""

from .execution_capsule import ExecutionCapsule
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


class CapsuleManager:
    """Manages execution capsules and environment tracking."""

    def __init__(self, capsule_dir: Path):
        self.capsule_dir = Path(capsule_dir)
        self.capsule_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.capsule_dir / "capsule_index.json"
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load capsule index."""
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                return json.load(f)
        return {"capsules": [], "environments": {}}

    def _save_index(self) -> None:
        """Save capsule index."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def create_capsule(self, session_id: str) -> ExecutionCapsule:
        """Create a new execution capsule."""
        capsule = ExecutionCapsule(session_id=session_id)
        return capsule

    def register_capsule(self, capsule: ExecutionCapsule) -> None:
        """Register a capsule in the index."""
        entry = {
            "capsule_id": capsule.capsule_id,
            "session_id": capsule.session_id,
            "created_at": capsule.manifest.created_at,
            "environment_hash": capsule.manifest.environment_hash,
        }
        self._index["capsules"].append(entry)

        # Track environment
        env_hash = capsule.manifest.environment_hash
        if env_hash not in self._index["environments"]:
            self._index["environments"][env_hash] = {
                "first_seen": capsule.manifest.created_at,
                "capsule_count": 0,
            }
        self._index["environments"][env_hash]["capsule_count"] += 1

        self._save_index()

    def find_matching_environment(self, capsule: ExecutionCapsule) -> Optional[ExecutionCapsule]:
        """Find a capsule with matching environment."""
        target_hash = capsule.manifest.environment_hash

        for entry in self._index["capsules"]:
            if entry["environment_hash"] == target_hash:
                manifest_path = self.capsule_dir / entry["session_id"] / "capsule_manifest.json"
                if manifest_path.exists():
                    return ExecutionCapsule.load(manifest_path)

        return None

    def get_environment_history(self) -> List[Dict[str, Any]]:
        """Get history of environments seen."""
        return [{"hash": h, **info} for h, info in self._index["environments"].items()]
