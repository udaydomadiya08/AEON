"""
AEON Phase 9 — Workload Registry.

Provides registration, retrieval, filtering, and cryptographic hashing
for all evaluation workloads.
"""

import hashlib
import json
from typing import Dict, List, Optional
from aeon.benchmarks.types import Workload, WorkloadFamily
from aeon.benchmarks.workloads import create_canonical_workloads


class WorkloadRegistry:
    """Registry managing benchmark workloads."""

    def __init__(self):
        self._workloads: Dict[str, Workload] = {}
        self.load_canonical()

    def register(self, workload: Workload) -> None:
        """Register a workload instance."""
        self._workloads[workload.id] = workload

    def get(self, workload_id: str) -> Optional[Workload]:
        """Retrieve workload by unique ID."""
        return self._workloads.get(workload_id)

    def list_workloads(self, family: Optional[WorkloadFamily] = None) -> List[Workload]:
        """List all workloads or filter by workload family."""
        if family is None:
            return list(self._workloads.values())
        return [w for w in self._workloads.values() if w.family == family]

    def load_canonical(self) -> None:
        """Load the standard canonical workload suite."""
        canonical = create_canonical_workloads()
        for w in canonical:
            self.register(w)

    def compute_suite_hash(self, workloads: Optional[List[Workload]] = None) -> str:
        """Generate a SHA-256 cryptographic digest of the workload set."""
        target = workloads if workloads is not None else list(self._workloads.values())
        serialized = []
        for w in sorted(target, key=lambda x: x.id):
            serialized.append({
                "id": w.id,
                "query": w.query,
                "context": w.context,
                "family": w.family.value,
                "expected": w.expected_answer or str(w.expected_numeric_value),
                "difficulty": w.declared_difficulty
            })
        raw = json.dumps(serialized, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


global_workload_registry = WorkloadRegistry()
