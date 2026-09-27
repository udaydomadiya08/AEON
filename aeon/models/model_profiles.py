"""
Standard Model Profiles for AEON.

Economic, architectural, and compute performance parameters calibrated from
published specifications (Llama 3, DeepSeek, Phi-3, Mistral, and Jev).
"""

from aeon.types import ModelProfile

DECISION_MODEL_PROFILE = ModelProfile(
    name="aeon-decision-jev",
    param_count_b=0.08,
    precision="INT8",
    cost_per_1m_input_usd=0.02,
    cost_per_1m_output_usd=0.02,
    base_latency_ms=5.0,
    tokens_per_second=1500.0,
    quality_score=0.82,
    min_memory_mb=120.0,
    flops_per_token=1.6e8,  # 2 * 0.08B
    is_decision_model=True
)

SMALL_MODEL_PROFILE = ModelProfile(
    name="aeon-small-1.5b",
    param_count_b=1.5,
    precision="INT8",
    cost_per_1m_input_usd=0.15,
    cost_per_1m_output_usd=0.30,
    base_latency_ms=25.0,
    tokens_per_second=220.0,
    quality_score=0.86,
    min_memory_mb=1800.0,
    flops_per_token=3.0e9,  # 2 * 1.5B
    is_decision_model=False
)

MEDIUM_MODEL_PROFILE = ModelProfile(
    name="aeon-medium-7b",
    param_count_b=7.0,
    precision="FP8",
    cost_per_1m_input_usd=0.50,
    cost_per_1m_output_usd=1.00,
    base_latency_ms=75.0,
    tokens_per_second=90.0,
    quality_score=0.92,
    min_memory_mb=8000.0,
    flops_per_token=1.4e10,  # 2 * 7B
    is_decision_model=False
)

LARGE_MODEL_PROFILE = ModelProfile(
    name="aeon-large-70b",
    param_count_b=70.0,
    precision="FP8",
    cost_per_1m_input_usd=2.50,
    cost_per_1m_output_usd=5.00,
    base_latency_ms=280.0,
    tokens_per_second=28.0,
    quality_score=0.97,
    min_memory_mb=72000.0,
    flops_per_token=1.4e11,  # 2 * 70B
    is_decision_model=False
)

DEFAULT_PROFILES = {
    DECISION_MODEL_PROFILE.name: DECISION_MODEL_PROFILE,
    SMALL_MODEL_PROFILE.name: SMALL_MODEL_PROFILE,
    MEDIUM_MODEL_PROFILE.name: MEDIUM_MODEL_PROFILE,
    LARGE_MODEL_PROFILE.name: LARGE_MODEL_PROFILE,
}
