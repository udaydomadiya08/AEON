"""
Safe AST-Based Deterministic Calculator for AEON.

Evaluates mathematical expressions without invoking dangerous eval() or arbitrary code execution.
Conforms to AEON Specification Section 9.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, Dict, Optional, Set
from aeon.types import TaskType
from aeon.tools.base import BaseTool
from aeon.tools.types import (
    ToolType,
    ToolCategory,
    ToolCapability,
    ToolRequest,
    ToolResult,
    ToolCost,
    ToolStatus,
    VerificationResult,
    VerificationStatus,
)


class SafeCalculator(BaseTool):
    """
    Deterministic offline mathematical calculator using AST-based safe evaluation.
    Rejects attribute access, imports, arbitrary function calls, and file/network operations.
    """

    name = "calculator"
    tool_type = ToolType.CALCULATOR
    category = ToolCategory.DETERMINISTIC

    # Safe mathematical functions and constants
    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    SAFE_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    PERCENT_OF_PATTERN = re.compile(r"([\d\.]+)\s*%\s*(?:of)\s*([\d\.,]+)", re.IGNORECASE)
    ARITHMETIC_PREFIX_PATTERN = re.compile(r"(?:what\s+is|calculate|evaluate|compute)\s+(.*)", re.IGNORECASE)

    def extract_expression(self, text: str) -> Optional[str]:
        """Extract or normalize an arithmetic expression from natural language."""
        text = text.strip()
        
        # 1. Percentage check: e.g. "17.5% of 840" or "What is 17.5% of 840?"
        pct_match = self.PERCENT_OF_PATTERN.search(text)
        if pct_match:
            pct_val = pct_match.group(1)
            base_val = pct_match.group(2).replace(",", "")
            return f"({pct_val} / 100.0) * {base_val}"

        # 2. Strip leading conversational prefix
        clean = text
        m = self.ARITHMETIC_PREFIX_PATTERN.match(text)
        if m:
            clean = m.group(1)

        # Remove trailing question marks or punctuation
        clean = clean.rstrip("?").rstrip(".").strip()

        # Check if contains arithmetic symbols
        if any(c in clean for c in ["+", "-", "*", "/", "%", "^"]):
            # Normalize '^' to '**' if used as power
            clean = clean.replace("^", "**")
            return clean

        # Simple standalone numbers
        try:
            float(clean)
            return clean
        except ValueError:
            return None

    def can_handle(self, request: ToolRequest) -> Optional[ToolCapability]:
        """
        Capability detector: Checks if request contains an evaluable arithmetic expression.
        Rejects symbolic equations or multi-variable problems that require SymPy/Reasoning.
        """
        # If query has variables or equation symbols like '=' (e.g. solve x^2 + 5x + 6 = 0), calculator cannot handle
        if "=" in request.query or "solve " in request.query.lower():
            return None

        expr = self.extract_expression(request.query)
        if expr is None:
            return None

        # Verify AST parse safety
        try:
            node = ast.parse(expr, mode="eval")
            if not self._is_ast_safe(node):
                return None
        except Exception:
            return None

        return ToolCapability(
            tool_type=ToolType.CALCULATOR,
            category=ToolCategory.DETERMINISTIC,
            supported_task_types=[TaskType.MATH, TaskType.ARITHMETIC],
            supported_operations=["addition", "subtraction", "multiplication", "division", "modulo", "power", "percentage"],
            deterministic=True,
            requires_network=False,
            requires_sandbox=False,
            expected_quality=1.0,
            expected_latency_ms=0.08,
            expected_cost_usd=0.0000005,
            expected_compute_flops=5e4,
            expected_memory_mb=0.1,
            expected_energy_joules=0.000002,
            risk_level=0.001,
            confidence=0.999,
            execution_mode="REAL_EXECUTION",
        )

    def _is_ast_safe(self, node: ast.AST) -> bool:
        """Strictly validate that only allowed AST nodes exist."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load)):
                continue
            elif isinstance(child, ast.Name):
                if child.id not in self.SAFE_CONSTANTS and child.id not in self.SAFE_FUNCTIONS:
                    return False
            elif isinstance(child, ast.Call):
                if not isinstance(child.func, ast.Name) or child.func.id not in self.SAFE_FUNCTIONS:
                    return False
            elif type(child) in self.SAFE_OPERATORS:
                continue
            else:
                return False
        return True

    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate an approved AST node."""
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in self.SAFE_CONSTANTS:
                return self.SAFE_CONSTANTS[node.id]
            raise ValueError(f"Unknown constant: {node.id}")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self.SAFE_OPERATORS:
                operand = self._eval_node(node.operand)
                return self.SAFE_OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.SAFE_OPERATORS:
                left = self._eval_node(node.left)
                right = self._eval_node(node.right)
                # Guard against extreme exponents to prevent DoS
                if op_type == ast.Pow and (right > 10000 or right < -10000):
                    raise ValueError("Exponent exceeds safe range")
                return self.SAFE_OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")
        elif isinstance(node, ast.Call):
            func_name = node.func.id  # type: ignore
            func = self.SAFE_FUNCTIONS[func_name]
            args = [self._eval_node(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"Unsupported AST node: {type(node)}")

    def execute(self, request: ToolRequest) -> ToolResult:
        """Execute safe mathematical evaluation."""
        expr = self.extract_expression(request.query)
        if expr is None:
            return ToolResult(
                tool_type=ToolType.CALCULATOR,
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error="Could not extract a valid mathematical expression from query",
                actual_cost=ToolCost(latency_ms=0.05, cost_usd=0.0000005, compute_flops=1e4),
                confidence=0.0,
            )

        try:
            tree = ast.parse(expr, mode="eval")
            if not self._is_ast_safe(tree):
                return ToolResult(
                    tool_type=ToolType.CALCULATOR,
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Expression contains forbidden AST nodes or unsafe operations",
                    actual_cost=ToolCost(latency_ms=0.05, cost_usd=0.0000005, compute_flops=1e4),
                    confidence=0.0,
                )

            val = self._eval_node(tree)
            # Format cleanly
            if isinstance(val, float) and val.is_integer():
                val_formatted = int(val)
            elif isinstance(val, float):
                val_formatted = round(val, 6)
            else:
                val_formatted = val

            return ToolResult(
                tool_type=ToolType.CALCULATOR,
                status=ToolStatus.SUCCESS,
                output=str(val_formatted),
                actual_cost=ToolCost(
                    latency_ms=0.08,
                    cost_usd=0.0000005,
                    compute_flops=5e4,
                    memory_mb=0.1,
                    energy_joules=0.000002,
                ),
                confidence=1.0,
                raw_metadata={"raw_result": val, "expression": expr},
            )
        except ZeroDivisionError:
            return ToolResult(
                tool_type=ToolType.CALCULATOR,
                status=ToolStatus.EXECUTION_ERROR,
                output=None,
                error="Division by zero",
                actual_cost=ToolCost(latency_ms=0.05, cost_usd=0.0000005),
                confidence=0.0,
            )
        except Exception as e:
            return ToolResult(
                tool_type=ToolType.CALCULATOR,
                status=ToolStatus.EXECUTION_ERROR,
                output=None,
                error=f"Calculator evaluation error: {str(e)}",
                actual_cost=ToolCost(latency_ms=0.05, cost_usd=0.0000005),
                confidence=0.0,
            )

    def verify(self, result: ToolResult) -> VerificationResult:
        """Verify calculator output is well-formed numerical result."""
        if result.status != ToolStatus.SUCCESS or result.output is None:
            return VerificationResult(
                status=VerificationStatus.VERIFICATION_FAILED,
                verified_output=None,
                verifier_used="calculator_verifier",
                discrepancy_detected=True,
                explanation=f"Execution failed: {result.error}",
            )
        try:
            float(result.output)
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                verified_output=result.output,
                verifier_used="calculator_verifier",
                discrepancy_detected=False,
                explanation="Numerical output valid and verified",
                verification_cost_usd=0.0000001,
                verification_latency_ms=0.01,
            )
        except ValueError:
            return VerificationResult(
                status=VerificationStatus.VERIFICATION_FAILED,
                verified_output=result.output,
                verifier_used="calculator_verifier",
                discrepancy_detected=True,
                explanation="Output could not be parsed as numerical",
            )
