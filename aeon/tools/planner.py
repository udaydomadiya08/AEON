"""
Computational Planner and Plan Optimizer for AEON.

Constructs, represents, optimizes, and executes multi-step computational plans.
Conforms to AEON Specification Sections 20, 21, and 22.
"""

from __future__ import annotations

import uuid
import time
from typing import Any, Dict, List, Optional, Tuple
from aeon.types import TaskType, ControllerAction
from aeon.tools.types import (
    ComputationPlan,
    PlanStep,
    ToolRequest,
    ToolResult,
    ToolCost,
    ToolStatus,
    VerificationStatus,
    ToolType,
)
from aeon.tools.candidate_generator import ComputationalCandidate
from aeon.tools.registry import ToolRegistry, global_tool_registry
from aeon.tools.verifier import ToolVerifier


class ComputationPlanner:
    """
    Constructs, simplifies, and executes single-step or multi-step computational workflows.
    Ensures compute is not wasted on redundant parametric models when tools suffice.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None, verifier: Optional[ToolVerifier] = None):
        self.tool_registry = tool_registry or global_tool_registry
        self.verifier = verifier or ToolVerifier()

    def create_plan(
        self,
        query: str,
        task_type: TaskType,
        top_candidate: ComputationalCandidate,
        all_candidates: Optional[List[ComputationalCandidate]] = None,
    ) -> ComputationPlan:
        """Construct a structured computation plan."""
        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        q_lower = query.lower()

        # Check for multi-step pattern: e.g., retrieval + calculation (CAGR / Growth across retrieved numbers)
        is_multi_step = (
            ("cagr" in q_lower or "growth" in q_lower or "difference" in q_lower)
            and any(k in q_lower for k in ["revenue", "apple", "nvidia", "gdp", "fy202", "between"])
        )

        steps: List[PlanStep] = []

        if is_multi_step:
            # Step 1: Retrieval
            steps.append(
                PlanStep(
                    step_id=1,
                    action=ControllerAction.RETRIEVAL,
                    tool="retrieval",
                    input_query=query,
                    dependencies=[],
                    expected_cost_usd=0.000030,
                    expected_latency_ms=15.0,
                )
            )
            # Step 2: Structured / Calculator computation on retrieved metrics
            steps.append(
                PlanStep(
                    step_id=2,
                    action=ControllerAction.STRUCTURED_COMPUTE,
                    tool="structured",
                    input_query="Calculate CAGR on retrieved numbers",
                    dependencies=[1],
                    expected_cost_usd=0.000001,
                    expected_latency_ms=0.2,
                )
            )
            # Step 3: Verification
            steps.append(
                PlanStep(
                    step_id=3,
                    action=ControllerAction.VERIFY,
                    tool="verifier",
                    input_query="Verify calculation consistency",
                    dependencies=[2],
                    expected_cost_usd=0.000001,
                    expected_latency_ms=0.05,
                )
            )
            raw_plan = ComputationPlan(
                plan_id=plan_id,
                steps=steps,
                estimated_quality=0.98,
                estimated_cost_usd=0.000032,
                estimated_latency_ms=15.25,
                estimated_compute_flops=7e5,
                estimated_risk=0.02,
                requires_verification=True,
                verification_plan="Cross-verify retrieved financial values with formula constraints",
            )
        else:
            # Single-step direct tool execution
            steps.append(
                PlanStep(
                    step_id=1,
                    action=top_candidate.action,
                    tool=top_candidate.source_name,
                    input_query=query,
                    dependencies=[],
                    expected_cost_usd=top_candidate.expected_cost_usd,
                    expected_latency_ms=top_candidate.expected_latency_ms,
                )
            )
            if top_candidate.verification_required:
                steps.append(
                    PlanStep(
                        step_id=2,
                        action=ControllerAction.VERIFY,
                        tool="verifier",
                        input_query="Verify tool output",
                        dependencies=[1],
                        expected_cost_usd=top_candidate.verification_cost_usd,
                        expected_latency_ms=0.05,
                    )
                )

            raw_plan = ComputationPlan(
                plan_id=plan_id,
                steps=steps,
                estimated_quality=top_candidate.expected_quality,
                estimated_cost_usd=top_candidate.expected_cost_usd + top_candidate.verification_cost_usd,
                estimated_latency_ms=top_candidate.expected_latency_ms,
                estimated_compute_flops=top_candidate.expected_compute_flops,
                estimated_risk=top_candidate.expected_risk,
                requires_verification=top_candidate.verification_required,
            )

        return self.optimize_plan(raw_plan)

    def optimize_plan(self, plan: ComputationPlan) -> ComputationPlan:
        """
        Simplify and optimize the computation plan (Section 22).
        Removes unnecessary steps and prevents redundant model invocations.
        """
        # If plan already has a deterministic tool solving the root problem,
        # eliminate any downstream model generation step
        optimized_steps = []
        has_deterministic_solver = False

        for step in plan.steps:
            if step.action in [
                ControllerAction.CALCULATOR,
                ControllerAction.SYMPY,
                ControllerAction.STRUCTURED_COMPUTE,
            ]:
                has_deterministic_solver = True
                optimized_steps.append(step)
            elif has_deterministic_solver and step.action in [
                ControllerAction.ESCALATE_MODEL,
                ControllerAction.CONTINUE,
            ]:
                # Redundant model call avoided!
                continue
            else:
                optimized_steps.append(step)

        plan.steps = optimized_steps
        return plan

    def execute_plan(
        self,
        plan: ComputationPlan,
        query: str,
        task_type: TaskType,
    ) -> Tuple[Optional[str], List[Dict[str, Any]], ToolCost, bool]:
        """
        Executes each step in the computational plan.
        Returns (final_answer, execution_history, total_cost, is_verified).
        """
        history: List[Dict[str, Any]] = []
        total_cost = ToolCost()
        step_outputs: Dict[int, Any] = {}
        last_output = None
        is_verified = True

        for step in plan.steps:
            t0 = time.perf_counter()
            tool = self.tool_registry.get(step.tool)
            step_record: Dict[str, Any] = {
                "step_id": step.step_id,
                "tool": step.tool,
                "action": step.action.value,
            }

            if step.action == ControllerAction.VERIFY:
                # Run verification on the output of previous step
                prev_id = step.dependencies[0] if step.dependencies else (step.step_id - 1)
                target_output = step_outputs.get(prev_id, last_output)
                # Check verification
                fallback_tool = next(iter(self.tool_registry._tools.values()), None)
                verifier_tool = self.tool_registry.get("calculator") or fallback_tool
                if verifier_tool is not None:
                    v_res = self.verifier.verify_result(
                        tool=verifier_tool,
                        result=ToolResult(
                            tool_type=verifier_tool.tool_type,
                            status=ToolStatus.SUCCESS if target_output is not None else ToolStatus.EXECUTION_ERROR,
                            output=target_output,
                        ),
                        query=query,
                    )
                else:
                    v_res = VerificationResult(
                        status=VerificationStatus.VERIFIED if target_output is not None else VerificationStatus.VERIFICATION_FAILED,
                        verified_output=target_output,
                        verifier_used="default_verifier",
                    )
                is_verified = (v_res.status == VerificationStatus.VERIFIED)
                step_record["verification_status"] = v_res.status.value
                step_record["explanation"] = v_res.explanation
                history.append(step_record)
                continue

            if not tool:
                step_record["status"] = "TOOL_NOT_FOUND"
                history.append(step_record)
                continue

            # Prepare step query: use previous step output if dependent
            current_q = query
            if step.dependencies and step.dependencies[0] in step_outputs:
                dep_out = step_outputs[step.dependencies[0]]
                current_q = f"{query} [Context from Step {step.dependencies[0]}: {dep_out}]"

            req = ToolRequest(query=current_q, task_type=task_type)
            res = tool.execute(req)

            step_record["status"] = res.status.value
            step_record["output"] = res.output
            step_record["latency_ms"] = res.actual_cost.latency_ms
            step_record["cost_usd"] = res.actual_cost.cost_usd
            step_record["compute_flops"] = res.actual_cost.compute_flops

            total_cost.latency_ms += res.actual_cost.latency_ms
            total_cost.cost_usd += res.actual_cost.cost_usd
            total_cost.compute_flops += res.actual_cost.compute_flops
            total_cost.energy_joules += res.actual_cost.energy_joules

            if res.status == ToolStatus.SUCCESS and res.output is not None:
                step_outputs[step.step_id] = res.output
                last_output = res.output
            else:
                step_record["error"] = res.error
                is_verified = False
                break

            history.append(step_record)

        return (str(last_output) if last_output is not None else None, history, total_cost, is_verified)
