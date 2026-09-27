"""
Tool Candidate Generator for AEON.

Evaluates capabilities across available tools and models to generate
viable computational candidates ranked by formal multi-objective utility.
Conforms to AEON Specification Sections 6, 7, and 8.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from aeon.types import (
    TaskType,
    ControllerAction,
    UtilityWeights,
    BudgetConstraints,
    RouteChoice,
)
from aeon.tools.types import (
    ToolType,
    ToolCapability,
    ToolRequest,
)
from aeon.tools.registry import ToolRegistry, global_tool_registry
from aeon.models.model_registry import ModelRegistry, global_registry


class ComputationalCandidate(BaseModel):
    """A viable computational path (tool or model) evaluated for economic selection."""
    action: ControllerAction
    source_name: str  # Tool name or model name
    tool_capability: Optional[ToolCapability] = None
    expected_quality: float
    expected_latency_ms: float
    expected_cost_usd: float
    expected_compute_flops: float
    expected_memory_mb: float
    expected_energy_joules: float
    expected_risk: float
    verification_required: bool = False
    verification_cost_usd: float = 0.0
    net_utility: float = 0.0
    voi_ratio: float = 0.0
    reason: str = ""


class ToolCandidateGenerator:
    """
    Generates and economically evaluates candidate computational actions
    (Tools vs Models) for an incoming request.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        model_registry: Optional[ModelRegistry] = None,
        weights: Optional[UtilityWeights] = None,
    ):
        self.tool_registry = tool_registry or global_tool_registry
        self.model_registry = model_registry or global_registry
        self.weights = weights or UtilityWeights()

    def generate_candidates(
        self,
        query: str,
        task_type: TaskType,
        difficulty: float,
        uncertainty: float,
        budget: Optional[BudgetConstraints] = None,
        memory_hit: bool = False,
    ) -> List[ComputationalCandidate]:
        """
        Produce a list of viable computational candidates (both tools and models),
        calculate their multi-objective utility, and rank them by net utility.
        """
        candidates: List[ComputationalCandidate] = []
        req = ToolRequest(query=query, task_type=task_type)
        tool_caps = self.tool_registry.get_all_capabilities(req)

        # 1. Evaluate tool candidates
        for tool_name, cap in tool_caps.items():
            # Map tool to action
            if cap.tool_type == ToolType.CALCULATOR:
                action = ControllerAction.CALCULATOR
            elif cap.tool_type == ToolType.SYMPY:
                action = ControllerAction.SYMPY
            elif cap.tool_type == ToolType.PYTHON:
                action = ControllerAction.PYTHON
            elif cap.tool_type == ToolType.RETRIEVAL:
                action = ControllerAction.RETRIEVAL
            elif cap.tool_type == ToolType.STRUCTURED:
                action = ControllerAction.STRUCTURED_COMPUTE
            else:
                action = ControllerAction.USE_TOOL

            # Verification cost for tools
            v_req = cap.deterministic is False or uncertainty > 0.4
            v_cost = 0.000002 if v_req else 0.0

            net_u = self._compute_utility(
                quality=cap.expected_quality,
                latency_ms=cap.expected_latency_ms,
                cost_usd=cap.expected_cost_usd + v_cost,
                compute_flops=cap.expected_compute_flops,
                memory_mb=cap.expected_memory_mb,
                energy_joules=cap.expected_energy_joules,
                risk=cap.risk_level,
            )

            voi_ratio = (cap.expected_quality / cap.expected_cost_usd) if cap.expected_cost_usd > 0 else 9999.0

            candidates.append(
                ComputationalCandidate(
                    action=action,
                    source_name=tool_name,
                    tool_capability=cap,
                    expected_quality=cap.expected_quality,
                    expected_latency_ms=cap.expected_latency_ms,
                    expected_cost_usd=cap.expected_cost_usd,
                    expected_compute_flops=cap.expected_compute_flops,
                    expected_memory_mb=cap.expected_memory_mb,
                    expected_energy_joules=cap.expected_energy_joules,
                    expected_risk=cap.risk_level,
                    verification_required=v_req,
                    verification_cost_usd=v_cost,
                    net_utility=round(net_u, 5),
                    voi_ratio=round(voi_ratio, 2),
                    reason=f"Deterministic/Specialized tool '{tool_name}' available with capability {cap.supported_operations}",
                )
            )

        # 2. Add Baseline Model Candidates
        # Small Model
        sm_profile = self.model_registry.get("aeon-small-1.5b")
        sm_quality = max(0.20, sm_profile.quality_score - (difficulty * 0.45))
        sm_latency = sm_profile.base_latency_ms + 40.0
        sm_cost = (70 / 1_000_000.0) * sm_profile.cost_per_1m_output_usd
        sm_compute = 2.0 * 1.5e9 * 70
        sm_energy = (sm_latency / 1000.0) * 25.0
        sm_net_u = self._compute_utility(
            quality=sm_quality,
            latency_ms=sm_latency,
            cost_usd=sm_cost,
            compute_flops=sm_compute,
            memory_mb=sm_profile.min_memory_mb,
            energy_joules=sm_energy,
            risk=0.15 * uncertainty,
        )
        candidates.append(
            ComputationalCandidate(
                action=ControllerAction.CONTINUE,
                source_name="aeon-small-1.5b",
                expected_quality=round(sm_quality, 4),
                expected_latency_ms=round(sm_latency, 2),
                expected_cost_usd=round(sm_cost, 7),
                expected_compute_flops=sm_compute,
                expected_memory_mb=sm_profile.min_memory_mb,
                expected_energy_joules=round(sm_energy, 4),
                expected_risk=round(0.15 * uncertainty, 4),
                net_utility=round(sm_net_u, 5),
                voi_ratio=round(sm_quality / sm_cost, 2),
                reason="Parametric small model baseline",
            )
        )

        # Large Model
        lg_profile = self.model_registry.get("aeon-large-70b")
        lg_quality = max(0.40, lg_profile.quality_score - (difficulty * 0.15))
        lg_latency = lg_profile.base_latency_ms + 300.0
        lg_cost = (100 / 1_000_000.0) * lg_profile.cost_per_1m_output_usd
        lg_compute = 2.0 * 70e9 * 100
        lg_energy = (lg_latency / 1000.0) * 120.0
        lg_net_u = self._compute_utility(
            quality=lg_quality,
            latency_ms=lg_latency,
            cost_usd=lg_cost,
            compute_flops=lg_compute,
            memory_mb=lg_profile.min_memory_mb,
            energy_joules=lg_energy,
            risk=0.05 * uncertainty,
        )
        candidates.append(
            ComputationalCandidate(
                action=ControllerAction.ESCALATE_MODEL,
                source_name="aeon-large-70b",
                expected_quality=round(lg_quality, 4),
                expected_latency_ms=round(lg_latency, 2),
                expected_cost_usd=round(lg_cost, 7),
                expected_compute_flops=lg_compute,
                expected_memory_mb=lg_profile.min_memory_mb,
                expected_energy_joules=round(lg_energy, 4),
                expected_risk=round(0.05 * uncertainty, 4),
                net_utility=round(lg_net_u, 5),
                voi_ratio=round(lg_quality / lg_cost, 2),
                reason="High-capacity 70B dense model for complex reasoning",
            )
        )

        # 3. Filter candidates by Budget Constraints if provided
        filtered = []
        for cand in candidates:
            if budget:
                if budget.max_cost_usd is not None and cand.expected_cost_usd > budget.max_cost_usd:
                    continue
                if budget.max_latency_ms is not None and cand.expected_latency_ms > budget.max_latency_ms:
                    continue
            filtered.append(cand)

        if not filtered:
            filtered = candidates

        # 4. Sort by Net Utility (highest first)
        filtered.sort(key=lambda c: c.net_utility, reverse=True)
        return filtered

    def _compute_utility(
        self,
        quality: float,
        latency_ms: float,
        cost_usd: float,
        compute_flops: float,
        memory_mb: float,
        energy_joules: float,
        risk: float,
    ) -> float:
        """
        Unified AEON Utility Objective:
        U(a) = Q(a) - λ_lat * L(a) - λ_cost * C(a) - λ_comp * F(a) - λ_mem * M(a) - λ_energy * E(a) - λ_risk * R(a)
        """
        penalty = (
            self.weights.lambda_latency * latency_ms
            + self.weights.lambda_cost * cost_usd
            + self.weights.lambda_compute * compute_flops
            + self.weights.lambda_memory * memory_mb
            + self.weights.lambda_energy * energy_joules
            + 0.5 * risk
        )
        return (self.weights.quality * quality) - penalty
