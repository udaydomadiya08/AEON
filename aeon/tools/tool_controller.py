"""
Tool Controller and Execution Engine for AEON.

Coordinates tool candidate evaluation, adaptive selection, plan execution,
cross-verification, budget enforcement, and failure recovery.
Conforms to AEON Specification Sections 17, 18, 19, 20.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from aeon.types import (
    TaskType,
    ControllerAction,
    DecisionReason,
    Telemetry,
    BudgetConstraints,
)
from aeon.tools.types import (
    ToolType,
    ToolRequest,
    ToolResult,
    ToolStatus,
    ToolBudget,
    ToolStats,
    VerificationStatus,
    VerificationResult,
)
from aeon.tools.registry import ToolRegistry, global_tool_registry
from aeon.tools.candidate_generator import ToolCandidateGenerator, ComputationalCandidate
from aeon.tools.planner import ComputationPlanner
from aeon.tools.verifier import ToolVerifier

DEFAULT_STATS_PATH = os.path.expanduser("~/.aeon_tool_stats.json")


class ToolController:
    """
    Central orchestrator for Phase 4 tool-aware computation.
    Evaluates tools as first-class computational paths alongside models and memory.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        candidate_generator: Optional[ToolCandidateGenerator] = None,
        planner: Optional[ComputationPlanner] = None,
        verifier: Optional[ToolVerifier] = None,
        budget: Optional[ToolBudget] = None,
        stats_path: str = DEFAULT_STATS_PATH,
    ):
        self.tool_registry = tool_registry or global_tool_registry
        self.candidate_generator = candidate_generator or ToolCandidateGenerator(self.tool_registry)
        self.verifier = verifier or ToolVerifier()
        self.planner = planner or ComputationPlanner(self.tool_registry, self.verifier)
        self.budget = budget or ToolBudget()
        self.stats = ToolStats()
        self.stats_path = stats_path
        self._load_stats()

    def _load_stats(self) -> None:
        """Load persistent tool statistics from disk if available."""
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.stats = ToolStats(**data)
            except Exception:
                pass

    def _save_stats(self) -> None:
        """Save persistent tool statistics to disk."""
        try:
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(self.stats.model_dump(), f, indent=2)
        except Exception:
            pass

    def clear_stats(self) -> None:
        """Reset operational tool telemetry."""
        self.stats = ToolStats()
        self._save_stats()

    def evaluate_and_execute(
        self,
        query: str,
        task_type: TaskType,
        difficulty: float,
        uncertainty: float,
        budget_constraints: Optional[BudgetConstraints] = None,
        require_verification: bool = False,
    ) -> Optional[Tuple[str, ComputationalCandidate, List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Evaluate candidate tools vs models. If a tool path is economically optimal,
        execute it with verification and failure fallback recovery.
        
        Returns:
            (answer, selected_candidate, execution_history, telemetry_meta) or None (if model is preferred)
        """
        # 1. Generate and rank candidates
        candidates = self.candidate_generator.generate_candidates(
            query=query,
            task_type=task_type,
            difficulty=difficulty,
            uncertainty=uncertainty,
            budget=budget_constraints,
        )

        if not candidates:
            return None

        top_cand = candidates[0]

        # 2. Check if the top candidate is a Tool action (rather than a Model action)
        tool_actions = [
            ControllerAction.CALCULATOR,
            ControllerAction.SYMPY,
            ControllerAction.PYTHON,
            ControllerAction.RETRIEVAL,
            ControllerAction.STRUCTURED_COMPUTE,
            ControllerAction.USE_TOOL,
        ]

        if top_cand.action not in tool_actions:
            # Model is economically superior or necessary for reasoning
            return None

        # 3. Build & optimize computation plan
        plan = self.planner.create_plan(query, task_type, top_cand, candidates)

        # 4. Execute with Failure Fallback Loop (Section 17)
        answer, history, cost, is_verified = self.planner.execute_plan(plan, query, task_type)

        # Update stats
        self.stats.total_calls += 1
        self.stats.calls_by_tool[top_cand.source_name] = self.stats.calls_by_tool.get(top_cand.source_name, 0) + 1

        fallback_occurred = False
        if answer is None or not is_verified:
            # Failure occurred: try fallback candidates
            self.stats.failed_calls += 1
            self.stats.fallbacks_triggered += 1
            fallback_occurred = True

            # Attempt fallback to next tool candidate if one exists
            fallback_cands = [c for c in candidates[1:] if c.action in tool_actions]
            for fb_cand in fallback_cands:
                fb_tool = self.tool_registry.get(fb_cand.source_name)
                if fb_tool:
                    fb_req = ToolRequest(query=query, task_type=task_type)
                    fb_res = fb_tool.execute(fb_req)
                    if fb_res.status == ToolStatus.SUCCESS and fb_res.output is not None:
                        answer = str(fb_res.output)
                        top_cand = fb_cand
                        history.append({
                            "fallback_to": fb_cand.source_name,
                            "output": answer,
                            "status": "SUCCESS",
                        })
                        cost.cost_usd += fb_res.actual_cost.cost_usd
                        cost.latency_ms += fb_res.actual_cost.latency_ms
                        cost.compute_flops += fb_res.actual_cost.compute_flops
                        is_verified = True
                        break

            # If still failed, return None so controller can safely escalate to parametric model
            if answer is None:
                self._save_stats()
                return None
        else:
            self.stats.successful_calls += 1
            self.stats.unnecessary_models_avoided += 1
            # Record compute savings vs 7B model
            self.stats.total_compute_saved_flops += max(0.0, 5e11 - cost.compute_flops)
            self.stats.total_cost_saved_usd += max(0.0, 0.0003 - cost.cost_usd)
            self.stats.total_latency_saved_ms += max(0.0, 400.0 - cost.latency_ms)

        if require_verification or top_cand.verification_required:
            self.stats.verification_runs += 1

        self._save_stats()

        meta = {
            "tool_used": top_cand.source_name,
            "tool_category": top_cand.tool_capability.category.value if top_cand.tool_capability else "deterministic",
            "tool_cost_usd": cost.cost_usd,
            "tool_latency_ms": cost.latency_ms,
            "tool_compute_flops": cost.compute_flops,
            "verification_status": VerificationStatus.VERIFIED.value if is_verified else VerificationStatus.VERIFICATION_FAILED.value,
            "unnecessary_model_avoided": True,
            "fallback_occurred": fallback_occurred,
            "plan_steps": history,
        }

        return (answer, top_cand, history, meta)


global_tool_controller = ToolController()
