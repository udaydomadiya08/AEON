"""
AEON — Adaptive Economically-Optimized Network.
Maximum-Efficiency AI Inference Architecture.
"""

from typing import Union, Optional
from aeon.types import InferenceRequest, InferenceResult, ExecutionPlan, Telemetry
from aeon.controller.compute_controller import ComputeController
from aeon.models.model_registry import global_registry, ModelRegistry
from aeon.inference.backend_interface import BaseInferenceBackend
from aeon.inference.simulation_backend import SimulationBackend
from aeon.inference.local_backend import LocalTransformersBackend

__version__ = "0.1.0"


from typing import Union, Optional, Dict, Any
from aeon.types import (
    InferenceRequest,
    InferenceResult,
    ExecutionPlan,
    Telemetry,
    BudgetConstraints,
    ComputeState,
    VOIResult,
    ControllerAction,
    DecisionReason,
    TaskType,
    DifficultyLevel,
    RouteChoice,
    ReasoningMode,
    UtilityWeights,
)
from aeon.controller.compute_controller import ComputeController
from aeon.controller.dynamic_stopping import DynamicStoppingController
from aeon.controller.voi_estimator import VOIEstimator
from aeon.models.model_registry import global_registry, ModelRegistry
from aeon.inference.backend_interface import BaseInferenceBackend
from aeon.inference.simulation_backend import SimulationBackend
from aeon.inference.local_backend import LocalTransformersBackend

from aeon.memory.hierarchy import AEONMemoryHierarchy, global_memory_hierarchy
from aeon.memory.types import (
    MemoryLevel,
    MemoryRequest,
    MemoryLookupResult,
    MemoryEntry,
    MemoryStats,
    MemoryBudget,
    ReuseDecision,
    EvictionPolicy,
    FreshnessPolicy,
    MemoryScope,
)

from aeon.tools.tool_controller import ToolController, global_tool_controller
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
    ComputationPlan,
    PlanStep,
    ToolBudget,
    ToolStats,
)
from aeon.tools.calculator import SafeCalculator
from aeon.tools.python_tool import RestrictedPythonTool
from aeon.tools.sympy_tool import SymPyTool
from aeon.tools.retrieval_tool import RetrievalTool
from aeon.tools.structured_tool import StructuredComputationTool
from aeon.tools.verifier import ToolVerifier
from aeon.tools.registry import ToolRegistry, global_tool_registry

from aeon.speculation.controller import SpeculationController, global_speculation_controller
from aeon.speculation.types import (
    SpeculationStrategy,
    SpeculationAction,
    DraftTokenProposal,
    VerificationResult as SpeculationVerificationResult,
    BreakEvenAnalysis,
    SpeculationCandidate,
    SpeculationState,
    SpeculationConfig,
)
from aeon.speculation.economics import SpeculationEconomics
from aeon.speculation.acceptance_estimator import AdaptiveAcceptanceEstimator
from aeon.speculation.draft_optimizer import AdaptiveDraftLengthOptimizer
from aeon.speculation.ngram import NgramSpeculationEngine
from aeon.speculation.verifier import SpeculationVerifier
from aeon.speculation.strategies import SpeculativeStrategySelector

from aeon.reasoning.controller import ReasoningController, global_reasoning_controller
from aeon.reasoning.types import (
    ReasoningStrategy,
    ReasoningAction,
    ReasoningStopReason,
    ReasoningBudget,
    ReasoningState,
    ReasoningCandidate,
    ConfidenceEstimate,
    ReasoningConfig,
)
from aeon.reasoning.economics import ReasoningEconomics
from aeon.reasoning.strategies import ReasoningStrategySelector

from aeon.depth.controller import DepthController, global_depth_controller
from aeon.depth.types import (
    DepthStrategy,
    DepthExitDecision,
    TokenImportanceSignal,
    ModelDepthProfile,
    TokenComputeState,
    DepthExecutionState,
    DepthCandidate,
    DepthConfig,
)
from aeon.depth.profiles import ModelDepthRegistry, global_depth_registry, get_depth_profile, get_model_depth_profile
from aeon.depth.cost_model import DepthCostModel, LayerCostModel, global_depth_cost_model
from aeon.depth.token_importance import TokenImportanceEstimator, global_token_importance_estimator
from aeon.depth.capacity_controller import CapacityController, global_capacity_controller
from aeon.depth.mod_router import MoDRouter, global_mod_router
from aeon.depth.early_exit import EarlyExitEvaluator, global_early_exit_evaluator
from aeon.depth.economics import DepthEconomics, global_depth_economics
from aeon.depth.backend import BaseDepthBackend, SimulationDepthBackend, global_simulation_depth_backend, global_depth_backend

from aeon.hardware.controller import HardwareController, global_hardware_controller
from aeon.hardware.types import (
    DeviceType,
    PrecisionType,
    QuantizationMethod,
    CapabilityStatus,
    ThermalPressure,
    MemoryPressure,
    ExecutionMode,
    HardwareProfile,
    PrecisionProfile,
    QuantizationConfig,
    QuantizationResult,
    MemoryAssessment,
    ThermalState,
    EnergyEstimate,
    HardwareCandidate,
    HardwareDecision,
    HardwareConfig,
)
from aeon.hardware.detector import HardwareDetector
from aeon.hardware.profiles import (
    APPLE_SILICON_M_PROFILE,
    NVIDIA_A100_PROFILE,
    NVIDIA_H100_PROFILE,
    CPU_SERVER_PROFILE,
    estimate_model_memory_mb,
)
from aeon.hardware.precision import PrecisionManager
from aeon.hardware.quantization import QuantizationManager
from aeon.hardware.capabilities import HardwareCapabilityMatrix
from aeon.hardware.memory import HardwareMemoryManager
from aeon.hardware.cost_model import HardwareCostModel
from aeon.hardware.energy import EnergyEstimator
from aeon.hardware.thermal import ThermalMonitor
from aeon.hardware.backend import BaseHardwareBackend, SimulationHardwareBackend


__version__ = "0.8.0"


class AEON:
    """
    AEON primary API facade.
    Provides simple inference interface: aeon.run(query) -> InferenceResult
    with comprehensive telemetry, routing breakdown, and compute accounting.
    Supports Phase 1 rule-based routing, cascades, Phase 2 VOI dynamic stopping,
    Phase 3 multi-level memory hierarchy, Phase 4 tool-aware computation,
    and Phase 5 adaptive speculative inference.
    """

    def __init__(
        self,
        backend: Optional[BaseInferenceBackend] = None,
        policy: str = "rules",  # "rules" (Phase 1 default), "voi" (Phase 2), "cascade"
        use_cascade: bool = False,
        registry: Optional[ModelRegistry] = None,
        voi_threshold: float = 15.0,
        memory: bool = False,
        memory_policy: str = "balanced",
        memory_budget_mb: float = 512.0,
        memory_hierarchy: Optional[AEONMemoryHierarchy] = None,
        tools: bool = False,
        tool_controller: Optional[ToolController] = None,
        speculation: bool = False,
        speculation_controller: Optional[SpeculationController] = None,
        reasoning: bool = False,
        reasoning_controller: Optional[ReasoningController] = None,
        adaptive_depth: bool = False,
        depth: bool = False,
        depth_strategy: Optional[str] = None,
        max_depth: Optional[int] = None,
        min_depth: Optional[int] = None,
        depth_controller: Optional[DepthController] = None,
        hardware_aware: bool = False,
        hardware: bool = False,
        preferred_device: Optional[str] = None,
        preferred_precision: Optional[str] = None,
        preferred_quantization: Optional[str] = None,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        quantization: Optional[str] = None,
        hardware_controller: Optional[HardwareController] = None,
    ):
        self.registry = registry or global_registry
        self.backend = backend or SimulationBackend(self.registry)
        self.policy = "cascade" if use_cascade else policy
        self.voi_threshold = voi_threshold
        self.memory_policy = memory_policy
        self.memory_enabled = memory
        self.tools_enabled = tools
        self.speculation_enabled = speculation
        self.reasoning_enabled = reasoning
        self.depth_enabled = adaptive_depth or depth
        self.depth_strategy = depth_strategy
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.hardware_enabled = hardware_aware or hardware
        self.preferred_device = device or preferred_device
        self.preferred_precision = precision or preferred_precision
        self.preferred_quantization = quantization or preferred_quantization

        # Initialize or assign memory hierarchy
        if memory_hierarchy is not None:
            self.memory_hierarchy = memory_hierarchy
        else:
            budget_bytes = int(memory_budget_mb * 1024 * 1024)
            budget = MemoryBudget(
                max_total_bytes=budget_bytes,
                max_l1_bytes=int(budget_bytes * 0.35),
                max_l2_bytes=int(budget_bytes * 0.25),
                max_l3_bytes=int(budget_bytes * 0.20),
                max_l4_bytes=int(budget_bytes * 0.20),
                eviction_policy=EvictionPolicy.VALUE_AWARE if memory_policy == "balanced" else EvictionPolicy.LRU,
            )
            self.memory_hierarchy = AEONMemoryHierarchy(budget=budget)

        # Initialize or assign tool controller
        self.tool_controller = tool_controller or global_tool_controller

        # Initialize or assign speculation controller
        self.speculation_controller = speculation_controller or global_speculation_controller

        # Initialize or assign reasoning controller
        self.reasoning_controller = reasoning_controller or global_reasoning_controller

        # Initialize or assign depth controller
        self.depth_controller = depth_controller or global_depth_controller

        # Initialize or assign hardware controller
        self.hardware_controller = hardware_controller or global_hardware_controller

        self.voi_controller = DynamicStoppingController(
            backend=self.backend,
            registry=self.registry,
            voi_estimator=VOIEstimator(self.registry, voi_threshold=voi_threshold),
            memory_hierarchy=self.memory_hierarchy,
            tool_controller=self.tool_controller,
            tools_enabled=self.tools_enabled,
            speculation_controller=self.speculation_controller,
            speculation_enabled=self.speculation_enabled,
            reasoning_controller=self.reasoning_controller,
            reasoning_enabled=self.reasoning_enabled,
            depth_controller=self.depth_controller,
            depth_enabled=self.depth_enabled,
            hardware_controller=self.hardware_controller,
            hardware_enabled=self.hardware_enabled,
        )
        self.rules_controller = ComputeController(
            backend=self.backend,
            registry=self.registry,
            use_cascade=(self.policy == "cascade"),
            memory_hierarchy=self.memory_hierarchy,
            tool_controller=self.tool_controller,
            tools_enabled=self.tools_enabled,
            speculation_controller=self.speculation_controller,
            speculation_enabled=self.speculation_enabled,
            reasoning_controller=self.reasoning_controller,
            reasoning_enabled=self.reasoning_enabled,
            depth_controller=self.depth_controller,
            depth_enabled=self.depth_enabled,
            hardware_controller=self.hardware_controller,
            hardware_enabled=self.hardware_enabled,
        )

    @property
    def memory(self) -> AEONMemoryHierarchy:
        """Access the multi-level memory hierarchy controller."""
        return self.memory_hierarchy

    @property
    def tools(self) -> ToolController:
        """Access the tool-aware computation controller."""
        return self.tool_controller

    @property
    def speculation(self) -> SpeculationController:
        """Access the speculative and token-efficient inference controller."""
        return self.speculation_controller

    @property
    def reasoning(self) -> ReasoningController:
        """Access the adaptive latent & token-efficient reasoning controller."""
        return self.reasoning_controller

    @property
    def depth(self) -> DepthController:
        """Access the adaptive model depth & Mixture-of-Depths controller."""
        return self.depth_controller

    @property
    def hardware(self) -> HardwareController:
        """Access the hardware-aware precision & quantization controller."""
        return self.hardware_controller

    def run(
        self,
        query: Union[str, InferenceRequest],
        context: Optional[str] = None,
        sla_latency_ms: Optional[float] = 2000.0,
        cost_budget_usd: Optional[float] = 0.01,
        target_quality: Optional[float] = None,
        budget: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[str] = None,
        memory: Optional[bool] = None,
        session_id: Optional[str] = None,
        consistency: Optional[str] = None,
        tools: Optional[bool] = None,
        require_verification: Optional[bool] = None,
        speculation: Optional[bool] = None,
        speculation_strategy: Optional[str] = None,
        draft_model: Optional[str] = None,
        reasoning: Optional[bool] = None,
        reasoning_strategy: Optional[str] = None,
        reasoning_budget_tokens: Optional[int] = None,
        reasoning_budget_cost: Optional[float] = None,
        adaptive_depth: Optional[bool] = None,
        depth_strategy: Optional[str] = None,
        max_depth: Optional[int] = None,
        min_depth: Optional[int] = None,
        hardware_aware: Optional[bool] = None,
        hardware: Optional[bool] = None,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        quantization: Optional[str] = None,
        memory_budget_mb: Optional[float] = None,
    ) -> InferenceResult:
        """Run an inference request through the AEON economically-optimized engine."""
        # Parse budget dictionary if provided
        budget_constraints = None
        if budget:
            budget_constraints = BudgetConstraints(
                max_cost_usd=budget.get("max_cost", budget.get("max_cost_usd", cost_budget_usd)),
                max_latency_ms=budget.get("max_latency", budget.get("max_latency_ms", sla_latency_ms)),
                target_quality=budget.get("target_quality", target_quality),
            )

        active_memory = memory if memory is not None else self.memory_enabled
        active_tools = tools if tools is not None else self.tools_enabled
        active_speculation = speculation if speculation is not None else self.speculation_enabled
        active_reasoning = reasoning if reasoning is not None else self.reasoning_enabled
        active_depth = adaptive_depth if adaptive_depth is not None else self.depth_enabled
        active_hardware = (hardware_aware or hardware) if (hardware_aware is not None or hardware is not None) else self.hardware_enabled

        if isinstance(query, str):
            req = InferenceRequest(
                query=query,
                context=context,
                sla_latency_ms=sla_latency_ms,
                cost_budget_usd=cost_budget_usd,
                target_quality=target_quality,
                ground_truth=ground_truth,
                budget_constraints=budget_constraints,
                memory=active_memory,
                session_id=session_id,
                consistency=consistency,
                tools=active_tools,
                require_verification=require_verification,
                speculation=active_speculation,
                speculation_strategy=speculation_strategy,
                draft_model=draft_model,
                reasoning=active_reasoning,
                reasoning_strategy=reasoning_strategy,
                reasoning_budget_tokens=reasoning_budget_tokens,
                reasoning_budget_cost=reasoning_budget_cost,
                target_confidence=target_quality,
                adaptive_depth=active_depth,
                depth_strategy=depth_strategy,
                max_depth=max_depth,
                min_depth=min_depth,
                hardware_aware=active_hardware,
                device=device or self.preferred_device,
                precision=precision or self.preferred_precision,
                quantization=quantization or self.preferred_quantization,
                memory_budget_mb=memory_budget_mb,
            )
        else:
            req = query
            if budget_constraints and not req.budget_constraints:
                req.budget_constraints = budget_constraints
            if memory is not None:
                req.memory = memory
            elif req.memory is None and self.memory_enabled:
                req.memory = True
            if tools is not None:
                req.tools = tools
            elif req.tools is None and self.tools_enabled:
                req.tools = True
            if speculation is not None:
                req.speculation = speculation
            elif req.speculation is None and self.speculation_enabled:
                req.speculation = True
            if speculation_strategy and not req.speculation_strategy:
                req.speculation_strategy = speculation_strategy
            if draft_model and not req.draft_model:
                req.draft_model = draft_model
            if reasoning is not None:
                req.reasoning = reasoning
            elif req.reasoning is None and self.reasoning_enabled:
                req.reasoning = True
            if adaptive_depth is not None:
                req.adaptive_depth = adaptive_depth
            elif req.adaptive_depth is None and self.depth_enabled:
                req.adaptive_depth = True
            if depth_strategy and not req.depth_strategy:
                req.depth_strategy = depth_strategy
            if max_depth and not req.max_depth:
                req.max_depth = max_depth
            if min_depth and not req.min_depth:
                req.min_depth = min_depth
            if reasoning_strategy and not req.reasoning_strategy:
                req.reasoning_strategy = reasoning_strategy
            if reasoning_budget_tokens and not req.reasoning_budget_tokens:
                req.reasoning_budget_tokens = reasoning_budget_tokens
            if reasoning_budget_cost and not req.reasoning_budget_cost:
                req.reasoning_budget_cost = reasoning_budget_cost
            if hardware_aware is not None or hardware is not None:
                req.hardware_aware = hardware_aware or hardware
            elif req.hardware_aware is None and self.hardware_enabled:
                req.hardware_aware = True
            if device and not req.device:
                req.device = device
            if precision and not req.precision:
                req.precision = precision
            if quantization and not req.quantization:
                req.quantization = quantization
            if memory_budget_mb and not req.memory_budget_mb:
                req.memory_budget_mb = memory_budget_mb
            if require_verification is not None:
                req.require_verification = require_verification
            if session_id and not req.session_id:
                req.session_id = session_id
            if consistency and not req.consistency:
                req.consistency = consistency

        # Dispatch based on policy
        if self.policy == "voi":
            return self.voi_controller.run(req)
        else:
            return self.rules_controller.run(req)

    def plan(self, query: str) -> ExecutionPlan:
        """Inspect the structured execution plan for a query without executing it."""
        req = InferenceRequest(query=query)
        return self.rules_controller.plan_execution(req)


__all__ = [
    "AEON",
    "InferenceRequest",
    "InferenceResult",
    "ExecutionPlan",
    "Telemetry",
    "ComputeState",
    "VOIResult",
    "ControllerAction",
    "DecisionReason",
    "TaskType",
    "DifficultyLevel",
    "RouteChoice",
    "ReasoningMode",
    "BudgetConstraints",
    "ComputeController",
    "DynamicStoppingController",
    "VOIEstimator",
    "AEONMemoryHierarchy",
    "global_memory_hierarchy",
    "MemoryLevel",
    "MemoryRequest",
    "MemoryLookupResult",
    "MemoryEntry",
    "MemoryStats",
    "MemoryBudget",
    "ReuseDecision",
    "EvictionPolicy",
    "FreshnessPolicy",
    "MemoryScope",
    "ToolController",
    "global_tool_controller",
    "ToolType",
    "ToolCategory",
    "ToolStatus",
    "VerificationStatus",
    "ToolCost",
    "ToolCapability",
    "ToolRequest",
    "ToolResult",
    "VerificationResult",
    "ComputationPlan",
    "PlanStep",
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
    "global_registry",
    "SpeculationController",
    "global_speculation_controller",
    "SpeculationStrategy",
    "SpeculationAction",
    "DraftTokenProposal",
    "SpeculationVerificationResult",
    "BreakEvenAnalysis",
    "SpeculationCandidate",
    "SpeculationState",
    "SpeculationConfig",
    "SpeculationEconomics",
    "AdaptiveAcceptanceEstimator",
    "AdaptiveDraftLengthOptimizer",
    "NgramSpeculationEngine",
    "SpeculationVerifier",
    "SpeculativeStrategySelector",
    # Phase 6 Adaptive Reasoning
    "ReasoningController",
    "global_reasoning_controller",
    "ReasoningStrategy",
    "ReasoningAction",
    "ReasoningStopReason",
    "ReasoningBudget",
    "ReasoningState",
    "ReasoningCandidate",
    "ConfidenceEstimate",
    "ReasoningConfig",
    "ReasoningEconomics",
    "ReasoningStrategySelector",
    # Phase 7 Adaptive Depth & MoD
    "DepthController",
    "global_depth_controller",
    "DepthStrategy",
    "DepthExitDecision",
    "TokenImportanceSignal",
    "ModelDepthProfile",
    "TokenComputeState",
    "DepthExecutionState",
    "DepthCandidate",
    "DepthConfig",
    "ModelDepthRegistry",
    "global_depth_registry",
    "get_depth_profile",
    "get_model_depth_profile",
    "DepthCostModel",
    "LayerCostModel",
    "global_depth_cost_model",
    "TokenImportanceEstimator",
    "global_token_importance_estimator",
    "CapacityController",
    "global_capacity_controller",
    "MoDRouter",
    "global_mod_router",
    "EarlyExitEvaluator",
    "global_early_exit_evaluator",
    "DepthEconomics",
    "global_depth_economics",
    "BaseDepthBackend",
    "SimulationDepthBackend",
    # Phase 8 Hardware & Quantization
    "HardwareController",
    "global_hardware_controller",
    "DeviceType",
    "PrecisionType",
    "QuantizationMethod",
    "CapabilityStatus",
    "ThermalPressure",
    "MemoryPressure",
    "ExecutionMode",
    "HardwareProfile",
    "PrecisionProfile",
    "QuantizationConfig",
    "QuantizationResult",
    "MemoryAssessment",
    "ThermalState",
    "EnergyEstimate",
    "HardwareCandidate",
    "HardwareDecision",
    "HardwareConfig",
    "HardwareDetector",
    "APPLE_SILICON_M_PROFILE",
    "NVIDIA_A100_PROFILE",
    "NVIDIA_H100_PROFILE",
    "CPU_SERVER_PROFILE",
    "estimate_model_memory_mb",
    "PrecisionManager",
    "QuantizationManager",
    "HardwareCapabilityMatrix",
    "HardwareMemoryManager",
    "HardwareCostModel",
    "EnergyEstimator",
    "ThermalMonitor",
    "BaseHardwareBackend",
    "SimulationHardwareBackend",
]
