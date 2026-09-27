"""
Base Tool Interface and Protocol for AEON.

Defines the formal Tool protocol conforming to AEON Specification Section 3.
All computational tools (deterministic, symbolic, programmatic, retrieval) implement this interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional
from aeon.tools.types import (
    ToolType,
    ToolCategory,
    ToolCapability,
    ToolRequest,
    ToolResult,
    ToolCost,
    VerificationResult,
    VerificationStatus,
)


@runtime_checkable
class Tool(Protocol):
    """Formal protocol for AEON computational tools."""

    name: str
    tool_type: ToolType
    category: ToolCategory

    def can_handle(self, request: ToolRequest) -> Optional[ToolCapability]:
        """
        Determines whether the tool can handle the request.
        Returns ToolCapability if applicable, or None if the request is unsupported.
        """
        ...

    def estimate_cost(self, request: ToolRequest) -> ToolCost:
        """Estimates resource and monetary costs for executing this request."""
        ...

    def execute(self, request: ToolRequest) -> ToolResult:
        """Executes the tool computation and returns a structured ToolResult."""
        ...

    def verify(self, result: ToolResult) -> VerificationResult:
        """Performs validation/verification on the produced result."""
        ...


class BaseTool:
    """Convenience base class providing default cost estimation and verification."""

    name: str = "base_tool"
    tool_type: ToolType = ToolType.CALCULATOR
    category: ToolCategory = ToolCategory.DETERMINISTIC

    def can_handle(self, request: ToolRequest) -> Optional[ToolCapability]:
        raise NotImplementedError

    def estimate_cost(self, request: ToolRequest) -> ToolCost:
        return ToolCost(
            latency_ms=0.1,
            cost_usd=0.000001,
            compute_flops=1e5,
            memory_mb=1.0,
            energy_joules=0.00001,
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        raise NotImplementedError

    def verify(self, result: ToolResult) -> VerificationResult:
        return VerificationResult(
            status=VerificationStatus.VERIFIED if result.error is None else VerificationStatus.VERIFICATION_FAILED,
            verified_output=result.output,
            verifier_used=f"{self.name}_self_verify",
            discrepancy_detected=result.error is not None,
            explanation="Self-verification passed" if result.error is None else str(result.error),
            verification_cost_usd=0.0,
            verification_latency_ms=0.01,
        )
