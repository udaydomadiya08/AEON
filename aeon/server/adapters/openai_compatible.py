"""
AEON Phase 10 — OpenAI-Compatible Adapter.

Translates standard OpenAI Chat Completion requests into the multi-dimensional AEON
orchestration pipeline and maps execution results back to OpenAI response shapes
while enriching responses with optional x_aeon telemetry.
"""

import time
import json
import asyncio
from typing import Dict, Any, Tuple, Optional, AsyncGenerator
from aeon import AEON
from aeon.server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatChoiceMessage,
    UsageInfo,
    AeonTelemetryMetadata,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
)
from aeon.server.errors import (
    InvalidRequestError,
    ServiceUnavailableError
)


class OpenAICompatibleAdapter:
    """Bridges OpenAI Chat Completions requests to the full AEON inference engine."""

    def __init__(self, aeon_engine: AEON):
        self.aeon = aeon_engine

    def parse_messages(self, request: ChatCompletionRequest) -> Tuple[str, Optional[str]]:
        """
        Extract primary query and contextual preamble from messages.
        The last user message is the active query; system messages and prior turns become context.
        """
        if not request.messages:
            raise InvalidRequestError("messages list cannot be empty", param="messages")

        system_prompts = []
        conversation_history = []
        last_user_query = None

        for msg in request.messages:
            if msg.role == "system":
                system_prompts.append(msg.content)
            elif msg.role == "user":
                last_user_query = msg.content
                conversation_history.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                conversation_history.append(f"Assistant: {msg.content}")

        if not last_user_query:
            raise InvalidRequestError("At least one user message must be provided", param="messages")

        # Assemble context: system prompt + prior turns (excluding the final query)
        context_parts = []
        if system_prompts:
            context_parts.append("\n".join(system_prompts))
        if len(conversation_history) > 1:
            context_parts.append("\n".join(conversation_history[:-1]))

        context = "\n\n".join(context_parts) if context_parts else None
        return last_user_query, context

    def execute_request(self, request: ChatCompletionRequest, request_id: str) -> ChatCompletionResponse:
        """Execute a non-streaming Chat Completion request through AEON."""
        query, context = self.parse_messages(request)

        # Extract namespaced AEON policy overrides if supplied in extra_body
        policy = request.extra_body.aeon if (request.extra_body and request.extra_body.aeon) else None

        sla_ms = policy.latency_sla_ms if policy else None
        cost_budget = policy.cost_budget_usd if policy else None
        quality_target = policy.quality_target if policy else None
        memory_budget = policy.memory_budget_mb if policy else None
        device = policy.preferred_device if policy else None
        precision = policy.preferred_precision if policy else None
        quantization = policy.preferred_quantization if policy else None

        memory = policy.memory if (policy and policy.memory is not None) else True
        tools = policy.tools if (policy and policy.tools is not None) else True
        speculation = policy.speculation if (policy and policy.speculation is not None) else True
        reasoning = policy.reasoning if (policy and policy.reasoning is not None) else True
        adaptive_depth = policy.adaptive_depth if (policy and policy.adaptive_depth is not None) else True
        hardware_aware = policy.hardware_aware if (policy and policy.hardware_aware is not None) else True

        # Preflight memory check
        if hasattr(self.aeon, "hardware_controller") and self.aeon.hardware_controller:
            hw_ctrl = self.aeon.hardware_controller
            headroom = getattr(hw_ctrl.profile, "ram_available_mb", 4096.0) if hasattr(hw_ctrl, "profile") else 4096.0
            if memory_budget and memory_budget > headroom:
                raise ServiceUnavailableError(
                    f"Memory safety rejection: Requested budget ({memory_budget}MB) exceeds available headroom ({headroom:.1f}MB)."
                )

        # Run through the existing AEON engine
        t_start = time.perf_counter()
        res = self.aeon.run(
            query=query,
            context=context,
            sla_latency_ms=sla_ms,
            cost_budget_usd=cost_budget,
            target_quality=quality_target,
            memory_budget_mb=memory_budget,
            device=device,
            precision=precision,
            quantization=quantization,
            memory=memory,
            tools=tools,
            speculation=speculation,
            reasoning=reasoning,
            adaptive_depth=adaptive_depth,
            hardware_aware=hardware_aware,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        tel = res.trace
        in_tokens = tel.input_tokens or 0
        out_tokens = tel.output_tokens or 0
        total_tokens = in_tokens + out_tokens

        model_name = getattr(res.plan, "model", None)
        if hasattr(model_name, "value"):
            model_name = model_name.value
        model_str = str(model_name) if model_name else request.model

        # Build telemetry metadata
        aeon_meta = AeonTelemetryMetadata(
            request_id=request_id,
            selected_model=model_str,
            selected_device=getattr(tel, "hardware_device", "cpu") or "cpu",
            selected_precision=getattr(tel, "precision_selected", "fp16") or "fp16",
            selected_depth=getattr(tel, "actual_depth", 32) or 32,
            selected_strategy=getattr(res.plan, "route", "DIRECT") if hasattr(res, "plan") else "DIRECT",
            cache_hit=bool(getattr(tel, "memory_hit", False)),
            tool_used=getattr(tel, "tool_used", None),
            reasoning_strategy=getattr(tel, "reasoning_strategy", None),
            speculation_used=bool(getattr(tel, "speculative", False)),
            quality_estimate=float(getattr(tel, "quality_estimate", 1.0) or 1.0),
            latency_ms=round(res.latency_ms or elapsed_ms, 2),
            estimated_cost_usd=round(res.cost or 1e-6, 7),
            estimated_energy_joules=round(getattr(tel, "energy_joules", 0.05) or 0.05, 4),
            controller_overhead_ms=round(getattr(tel, "hardware_controller_overhead_ms", 0.03) or 0.03, 3),
            provenance="HYBRID"
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-aeon-{request_id}",
            object="chat.completion",
            created=int(time.time()),
            model=model_str,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatChoiceMessage(
                        role="assistant",
                        content=res.answer
                    ),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=in_tokens,
                completion_tokens=out_tokens,
                total_tokens=total_tokens
            ),
            x_aeon=aeon_meta
        )

    async def stream_request(self, request: ChatCompletionRequest, request_id: str) -> AsyncGenerator[str, None]:
        """
        Stream completion chunks via Server-Sent Events.
        AEON routing and decisions occur before emission; tokens are then streamed asynchronously.
        """
        # Execute AEON request
        response = self.execute_request(request, request_id)
        content = response.choices[0].message.content
        model_str = response.model
        created_time = response.created

        # Initial chunk: role definition
        chunk_init = ChatCompletionChunk(
            id=response.id,
            object="chat.completion.chunk",
            created=created_time,
            model=model_str,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant", content=""),
                    finish_reason=None
                )
            ],
            x_aeon=response.x_aeon
        )
        yield f"data: {chunk_init.model_dump_json()}\n\n"

        # Stream content in tokens/words
        words = content.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            chunk_token = ChatCompletionChunk(
                id=response.id,
                object="chat.completion.chunk",
                created=created_time,
                model=model_str,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=token),
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {chunk_token.model_dump_json()}\n\n"
            await asyncio.sleep(0.005)  # Realistic pacing

        # Final terminal chunk
        chunk_final = ChatCompletionChunk(
            id=response.id,
            object="chat.completion.chunk",
            created=created_time,
            model=model_str,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {chunk_final.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
