"""
AEON Phase 10 — Pydantic API Schemas.

Provides OpenAI-compatible schemas for chat completions, models, streaming chunks,
and cleanly namespaced AEON telemetry and policy extensions.
"""

import time
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator


# =========================================================================
# 1. AEON Policy & Telemetry Extensions
# =========================================================================

class AeonPolicyOverride(BaseModel):
    """Namespaced per-request optimization constraints and overrides."""
    quality_target: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum acceptable quality threshold (0.0 to 1.0)")
    latency_sla_ms: Optional[float] = Field(default=None, gt=0.0, description="Strict SLA latency limit in milliseconds")
    cost_budget_usd: Optional[float] = Field(default=None, gt=0.0, description="Maximum allowed monetary cost in USD")
    memory_budget_mb: Optional[float] = Field(default=None, gt=0.0, description="Maximum allowable memory footprint in MB")
    energy_budget_joules: Optional[float] = Field(default=None, gt=0.0, description="Maximum allowable energy consumption in Joules")

    preferred_device: Optional[str] = Field(default=None, description="Preferred hardware accelerator (e.g. cpu, mps, cuda)")
    preferred_precision: Optional[str] = Field(default=None, description="Preferred numerical precision (e.g. fp32, fp16, fp8, int8)")
    preferred_quantization: Optional[str] = Field(default=None, description="Preferred quantization method (e.g. none, int8, int4)")

    memory: Optional[bool] = Field(default=None, description="Toggle Phase 3 multi-tier memory hierarchy")
    tools: Optional[bool] = Field(default=None, description="Toggle Phase 4 deterministic tool routing")
    speculation: Optional[bool] = Field(default=None, description="Toggle Phase 5 speculative decoding")
    reasoning: Optional[bool] = Field(default=None, description="Toggle Phase 6 adaptive reasoning controller")
    adaptive_depth: Optional[bool] = Field(default=None, description="Toggle Phase 7 adaptive depth early exit")
    hardware_aware: Optional[bool] = Field(default=None, description="Toggle Phase 8 hardware-aware execution")


class ExtraBody(BaseModel):
    """Holder for optional namespaced extensions."""
    aeon: Optional[AeonPolicyOverride] = Field(default=None, description="AEON-specific orchestration controls")


class AeonTelemetryMetadata(BaseModel):
    """Non-intrusive metadata exposed via x_aeon without violating OpenAI schema compatibility."""
    request_id: str
    selected_model: str
    selected_device: str
    selected_precision: str
    selected_depth: int
    selected_strategy: str
    cache_hit: bool
    tool_used: Optional[str] = None
    reasoning_strategy: Optional[str] = None
    speculation_used: bool = False
    quality_estimate: float
    latency_ms: float
    estimated_cost_usd: float
    estimated_energy_joules: float
    controller_overhead_ms: float
    provenance: str = "HYBRID"


# =========================================================================
# 2. Chat Completions Request Models
# =========================================================================

class ChatMessage(BaseModel):
    """Standard conversational message."""
    role: Literal["system", "user", "assistant", "tool", "function"] = Field(description="Role of the message author")
    content: str = Field(description="The message contents")
    name: Optional[str] = Field(default=None, description="Optional author participant name")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request schema."""
    model: str = Field(default="aeon-auto", description="Model identifier (e.g. 'aeon-auto', 'aeon-small', 'aeon-medium')")
    messages: List[ChatMessage] = Field(min_length=1, description="List of messages comprising the conversation so far")
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    n: Optional[int] = Field(default=1, ge=1, le=1, description="Number of completions (currently 1 supported)")
    stream: Optional[bool] = Field(default=False, description="Whether to stream response chunks via Server-Sent Events")
    max_tokens: Optional[int] = Field(default=256, ge=1, le=8192, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(default=None, description="A unique identifier representing your end-user")
    extra_body: Optional[ExtraBody] = Field(default=None, description="Optional extra body for AEON configuration")


# =========================================================================
# 3. Chat Completions Response Models
# =========================================================================

class ChatChoiceMessage(BaseModel):
    """Output message generated by the model."""
    role: str = "assistant"
    content: str


class ChatChoice(BaseModel):
    """Individual completion choice."""
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    """Token accounting breakdown."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Standard OpenAI-compatible Chat Completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-aeon-{int(time.time()*1000)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo
    x_aeon: Optional[AeonTelemetryMetadata] = None


# =========================================================================
# 4. Streaming Chunks
# =========================================================================

class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Chunk emitted during streaming via SSE."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    x_aeon: Optional[AeonTelemetryMetadata] = None


# =========================================================================
# 5. Model Registry & List
# =========================================================================

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "aeon"
    capabilities: Optional[Dict[str, Any]] = None
    x_aeon_metadata: Optional[Dict[str, Any]] = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]


# =========================================================================
# 6. Health & Diagnostic Schemas
# =========================================================================

class ComponentHealth(BaseModel):
    status: str
    available: bool
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "0.10.0"
    components: Dict[str, ComponentHealth]


class VersionResponse(BaseModel):
    aeon_version: str = "0.10.0"
    api_version: str = "v1"
    git_commit: str
    python_version: str
    timestamp: str
