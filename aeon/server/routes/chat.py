"""
AEON Phase 10 — Chat Completions Route.

Implements POST /v1/chat/completions supporting standard OpenAI request formats,
synchronous responses, SSE streaming, and transparent telemetry capture.
"""

import time
from fastapi import APIRouter, Depends, Request
from starlette.responses import StreamingResponse
from aeon.server.schemas import ChatCompletionRequest, ChatCompletionResponse
from aeon.server.dependencies import (
    get_server_state,
    require_auth,
    check_rate_limits,
)
from aeon.server.errors import InvalidRequestError

router = APIRouter(prefix="/v1", tags=["Chat Completions"])


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(require_auth), Depends(check_rate_limits)],
    summary="Create chat completion",
    description="Generate responses through AEON's adaptive compute orchestration engine."
)
async def create_chat_completion(
    request_data: ChatCompletionRequest,
    raw_request: Request
):
    state = get_server_state(raw_request)
    config = state.config
    request_id = getattr(raw_request.state, "request_id", f"req-{int(time.time()*1000)}")

    # 1. Validation checks
    valid_models = {"aeon-auto", "aeon-small", "aeon-medium", "aeon-large", "aeon-small-1.5b", "aeon-medium-7b", "aeon-large-70b", "aeon-draft-0.5b", "aeon-decision-0.08b"}
    if hasattr(state.aeon, "registry") and state.aeon.registry:
        for m in state.aeon.registry.list_models():
            valid_models.add(m.name)
    if request_data.model not in valid_models:
        from aeon.server.errors import ResourceNotFoundError
        raise ResourceNotFoundError(
            f"Model '{request_data.model}' does not exist or is not available.",
            param="model"
        )

    if len(request_data.messages) > config.max_messages:
        raise InvalidRequestError(
            f"Message count ({len(request_data.messages)}) exceeds server limit of {config.max_messages}",
            param="messages"
        )
    if request_data.max_tokens and request_data.max_tokens > config.max_max_tokens:
        raise InvalidRequestError(
            f"max_tokens ({request_data.max_tokens}) exceeds maximum limit of {config.max_max_tokens}",
            param="max_tokens"
        )

    # 2. Concurrency Backpressure
    await state.backpressure.acquire()
    try:
        if request_data.stream:
            # Streaming SSE response
            async def event_generator():
                try:
                    async for event in state.adapter.stream_request(request_data, request_id):
                        yield event
                finally:
                    await state.backpressure.release()
                    # Record telemetry
                    if config.telemetry_enabled:
                        await state.telemetry_store.record_event({
                            "request_id": request_id,
                            "model": request_data.model,
                            "status_code": 200,
                            "stream": True,
                        })

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                }
            )

        # Non-streaming response
        t_start = time.perf_counter()
        response = state.adapter.execute_request(request_data, request_id)
        duration_ms = (time.perf_counter() - t_start) * 1000.0

        # Record telemetry
        if config.telemetry_enabled:
            x_aeon = response.x_aeon
            event_data = {
                "request_id": request_id,
                "model": response.model,
                "device": x_aeon.selected_device if x_aeon else "cpu",
                "precision": x_aeon.selected_precision if x_aeon else "fp16",
                "depth": x_aeon.selected_depth if x_aeon else 32,
                "latency_ms": x_aeon.latency_ms if x_aeon else duration_ms,
                "cost_usd": x_aeon.estimated_cost_usd if x_aeon else 0.0,
                "energy_joules": x_aeon.estimated_energy_joules if x_aeon else 0.0,
                "controller_overhead_ms": x_aeon.controller_overhead_ms if x_aeon else 0.0,
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
                "cache_hit": x_aeon.cache_hit if x_aeon else False,
                "tool_used": x_aeon.tool_used if x_aeon else None,
                "reasoning_strategy": x_aeon.reasoning_strategy if x_aeon else None,
                "speculation_used": x_aeon.speculation_used if x_aeon else False,
                "quality_estimate": x_aeon.quality_estimate if x_aeon else 1.0,
                "status_code": 200,
                "stream": False,
                "provenance": x_aeon.provenance if x_aeon else "HYBRID"
            }
            await state.telemetry_store.record_event(event_data)

        return response
    finally:
        if not request_data.stream:
            await state.backpressure.release()
