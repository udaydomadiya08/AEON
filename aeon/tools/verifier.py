"""
Tool Result Verification and Cross-Verification Engine for AEON.

Provides invariant checks, schema checks, source freshness validation,
and cross-computational verification (e.g. model math verified by SymPy/Calculator).
Conforms to AEON Specification Sections 15 and 16.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from aeon.types import ControllerAction
from aeon.tools.base import BaseTool
from aeon.tools.types import (
    ToolResult,
    ToolStatus,
    VerificationResult,
    VerificationStatus,
)


class ToolVerifier:
    """
    Unified verification engine evaluating correctness, schema invariants,
    and cross-tool consistency.
    """

    def __init__(self, verification_cost_weight: float = 1.0):
        self.verification_cost_weight = verification_cost_weight

    def verify_result(
        self,
        tool: BaseTool,
        result: ToolResult,
        query: str,
    ) -> VerificationResult:
        """Execute tool's native self-verification or fallback validation."""
        if result.status != ToolStatus.SUCCESS or result.output is None:
            return VerificationResult(
                status=VerificationStatus.VERIFICATION_FAILED,
                verified_output=None,
                verifier_used=f"{tool.name}_verifier",
                discrepancy_detected=True,
                explanation=f"Tool execution failed: {result.error}",
                verification_cost_usd=0.0,
                verification_latency_ms=0.01,
            )

        # Delegate to tool's native verification
        return tool.verify(result)

    def cross_verify(
        self,
        primary_tool: str,
        primary_output: str,
        secondary_tool: BaseTool,
        query: str,
    ) -> VerificationResult:
        """
        Cross-verify a result produced by one computational mechanism using an independent tool.
        Example: Output from model or Python verified by SymPy or Calculator.
        """
        t0 = time.perf_counter()
        from aeon.tools.types import ToolRequest

        # Prepare request for secondary tool
        req = ToolRequest(query=query)
        cap = secondary_tool.can_handle(req)
        if not cap:
            return VerificationResult(
                status=VerificationStatus.VERIFICATION_SKIPPED,
                verified_output=primary_output,
                verifier_used=secondary_tool.name,
                discrepancy_detected=False,
                explanation=f"Secondary tool {secondary_tool.name} cannot handle query for cross-verification",
                verification_cost_usd=0.0,
                verification_latency_ms=0.01,
            )

        sec_res = secondary_tool.execute(req)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if sec_res.status != ToolStatus.SUCCESS or sec_res.output is None:
            return VerificationResult(
                status=VerificationStatus.VERIFICATION_FAILED,
                verified_output=primary_output,
                verifier_used=secondary_tool.name,
                discrepancy_detected=True,
                explanation=f"Cross-verifier {secondary_tool.name} failed during evaluation: {sec_res.error}",
                verification_cost_usd=sec_res.actual_cost.cost_usd,
                verification_latency_ms=latency_ms,
            )

        # Compare outputs: clean strings or floats
        prim_str = str(primary_output).strip()
        sec_str = str(sec_res.output).strip()

        # Try float comparison if both are numerical
        is_match = False
        try:
            f_prim = float(prim_str.split()[0].replace(",", "").replace("%", ""))
            f_sec = float(sec_str.split()[0].replace(",", "").replace("%", ""))
            is_match = abs(f_prim - f_sec) < 1e-4
        except Exception:
            is_match = (prim_str.lower() in sec_str.lower()) or (sec_str.lower() in prim_str.lower())

        if is_match:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                verified_output=primary_output,
                verifier_used=secondary_tool.name,
                discrepancy_detected=False,
                explanation=f"Cross-verification succeeded between {primary_tool} and {secondary_tool.name}",
                verification_cost_usd=sec_res.actual_cost.cost_usd,
                verification_latency_ms=latency_ms,
            )
        else:
            return VerificationResult(
                status=VerificationStatus.VERIFICATION_FAILED,
                verified_output=primary_output,
                verifier_used=secondary_tool.name,
                discrepancy_detected=True,
                explanation=f"Discrepancy detected: {primary_tool} output '{prim_str}' != {secondary_tool.name} output '{sec_str}'",
                verification_cost_usd=sec_res.actual_cost.cost_usd,
                verification_latency_ms=latency_ms,
            )

    def is_verification_economically_justified(
        self,
        current_uncertainty: float,
        expected_error_cost: float,
        verification_cost_usd: float,
    ) -> bool:
        """
        Calculates whether verification yields positive net utility:
        U_verify = (Error_Reduction * Cost_of_Error) - Verification_Cost > 0.
        """
        expected_savings = current_uncertainty * expected_error_cost
        return (expected_savings * self.verification_cost_weight) > verification_cost_usd
