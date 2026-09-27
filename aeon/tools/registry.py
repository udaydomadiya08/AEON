"""
Tool Registry for AEON.

Central catalog registering and providing computational tools.
Conforms to AEON Specification Sections 3, 4, and 5.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from aeon.tools.base import BaseTool
from aeon.tools.types import ToolType, ToolCapability, ToolRequest
from aeon.tools.calculator import SafeCalculator
from aeon.tools.python_tool import RestrictedPythonTool
from aeon.tools.sympy_tool import SymPyTool
from aeon.tools.retrieval_tool import RetrievalTool
from aeon.tools.structured_tool import StructuredComputationTool


class ToolRegistry:
    """Registry maintaining available tools and their capabilities."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the built-in deterministic, symbolic, programmatic, and retrieval tools."""
        self.register(SafeCalculator())
        self.register(RestrictedPythonTool())
        self.register(SymPyTool())
        self.register(RetrievalTool())
        self.register(StructuredComputationTool())

    def register(self, tool: BaseTool) -> None:
        """Add a tool to the registry."""
        self._tools[tool.name] = tool
        self._tools[tool.tool_type.value] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve tool by name or type identifier."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List distinct registered tool names."""
        distinct = sorted(list({t.name for t in self._tools.values()}))
        return distinct

    def get_all_capabilities(self, request: ToolRequest) -> Dict[str, ToolCapability]:
        """Evaluate which registered tools can handle the given request."""
        capabilities: Dict[str, ToolCapability] = {}
        for tool_name in self.list_tools():
            tool = self._tools[tool_name]
            cap = tool.can_handle(request)
            if cap is not None:
                capabilities[tool_name] = cap
        return capabilities


global_tool_registry = ToolRegistry()
