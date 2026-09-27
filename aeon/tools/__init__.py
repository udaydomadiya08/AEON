"""
AEON Phase 4 Tool-Aware Computation Module.

Exports tools, registries, planners, verifiers, and the central ToolController.
"""

from aeon.tools.types import (
    ToolType,
    ToolCategory,
    ToolStatus,
    VerificationStatus,
    ToolCost,
    ToolCapability,
    ToolRequest,
    ToolResult,
    VerificationResult,
    PlanStep,
    ComputationPlan,
    ToolBudget,
    ToolStats,
)
from aeon.tools.base import Tool, BaseTool
from aeon.tools.calculator import SafeCalculator
from aeon.tools.python_tool import RestrictedPythonTool
from aeon.tools.sympy_tool import SymPyTool
from aeon.tools.retrieval_tool import RetrievalTool
from aeon.tools.structured_tool import StructuredComputationTool
from aeon.tools.verifier import ToolVerifier
from aeon.tools.registry import ToolRegistry, global_tool_registry
from aeon.tools.candidate_generator import ToolCandidateGenerator, ComputationalCandidate
from aeon.tools.planner import ComputationPlanner
from aeon.tools.tool_controller import ToolController, global_tool_controller

__all__ = [
    "Tool",
    "BaseTool",
    "ToolType",
    "ToolCategory",
    "ToolStatus",
    "VerificationStatus",
    "ToolCost",
    "ToolCapability",
    "ToolRequest",
    "ToolResult",
    "VerificationResult",
    "PlanStep",
    "ComputationPlan",
    "ToolBudget",
    "ToolStats",
    "SafeCalculator",
    "RestrictedPythonTool",
    "SymPyTool",
    "RetrievalTool",
    "StructuredComputationTool",
    "ToolVerifier",
    "ToolRegistry",
    "global_tool_registry",
    "ToolCandidateGenerator",
    "ComputationalCandidate",
    "ComputationPlanner",
    "ToolController",
    "global_tool_controller",
]
