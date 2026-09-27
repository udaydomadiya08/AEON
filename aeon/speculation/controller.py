"""
AEON Speculation Controller.
Coordinates economic strategy selection, drafting, parallel verification,
adaptive draft length, dynamic stopping, and failure recovery.
Conforms to AEON Phase 5 Specification Sections 2, 7, 8, 17, 19, and 25.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from aeon.types import (
    ModelProfile,
    TaskType,
    Telemetry,
    ControllerAction,
    DecisionReason,
)
from aeon.models.model_registry import ModelRegistry, global_registry
from aeon.speculation.types import (
    SpeculationStrategy,
    SpeculationAction,
    SpeculationState,
    SpeculationCandidate,
    SpeculationConfig,
    DraftTokenProposal,
    VerificationResult,
)
from aeon.speculation.economics import SpeculationEconomics
from aeon.speculation.acceptance_estimator import AdaptiveAcceptanceEstimator
from aeon.speculation.draft_optimizer import AdaptiveDraftLengthOptimizer
from aeon.speculation.ngram import NgramSpeculationEngine
from aeon.speculation.verifier import SpeculationVerifier
from aeon.speculation.strategies import SpeculativeStrategySelector


class SpeculationController:
    """
    Central controller orchestrating speculative and token-efficient inference for AEON.
    Treats speculative decoding as an economic allocation decision controlled by VOI.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        config: Optional[SpeculationConfig] = None,
        mode: str = "SIMULATED_SPECULATIVE_DECODING",
    ):
        self.registry = registry or global_registry
        self.config = config or SpeculationConfig()
        self.mode = mode
        
        self.economics = SpeculationEconomics(self.config)
        self.estimator = AdaptiveAcceptanceEstimator(ema_alpha=self.config.learning_rate)
        self.optimizer = AdaptiveDraftLengthOptimizer(self.economics, self.estimator, self.config)
        self.ngram_engine = NgramSpeculationEngine()
        self.verifier = SpeculationVerifier(mode=self.mode)
        self.selector = SpeculativeStrategySelector(
            registry=self.registry,
            economics=self.economics,
            estimator=self.estimator,
            optimizer=self.optimizer,
            config=self.config,
        )

    def execute_generation(
        self,
        target_model: str,
        query: str,
        context: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        requested_tokens: int = 50,
        preferred_strategy: Optional[SpeculationStrategy] = None,
        preferred_draft_model: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> Tuple[str, SpeculationState, Telemetry]:
        """
        Execute speculative or standard token generation loop.
        Returns: (generated_text, speculation_state, telemetry)
        """
        request_id = f"spec-{uuid.uuid4().hex[:8]}"
        target_profile = self.registry.get(target_model)
        if not target_profile:
            target_profile = self.registry.get("aeon-small-1.5b")

        # 1. Strategy Selection
        if preferred_strategy is not None:
            candidates = self.selector.evaluate_all_strategies(
                target_model=target_profile.name,
                query=query,
                task_type=task_type,
                tokens_to_generate=requested_tokens,
                preferred_draft_model=preferred_draft_model,
            )
            matching = [c for c in candidates if c.strategy == preferred_strategy]
            selected_cand = matching[0] if matching else candidates[0]
        else:
            selected_cand, candidates = self.selector.select_strategy(
                target_model=target_profile.name,
                query=query,
                task_type=task_type,
                tokens_to_generate=requested_tokens,
                preferred_draft_model=preferred_draft_model,
            )

        # Initialize State
        state = SpeculationState(
            request_id=request_id,
            target_model=target_profile.name,
            current_strategy=selected_cand.strategy,
            draft_model=selected_cand.draft_model,
            draft_length=selected_cand.draft_length,
            expected_speedup=selected_cand.expected_speedup,
            expected_utility=selected_cand.expected_utility,
            execution_mode=self.mode,
        )

        full_prompt = (context + " " + query) if context else query
        self.ngram_engine.index_context(full_prompt)

        # Reference target tokens to verify against (ground truth simulation)
        target_tokens = self._synthesize_target_sequence(full_prompt, requested_tokens, reference_text)
        
        generated_tokens: List[str] = []
        token_cursor = 0
        total_time_ms = 0.0

        target_token_time = 1000.0 / max(1.0, target_profile.tokens_per_second)

        # Standard Decoding Execution Path
        if state.current_strategy == SpeculationStrategy.STANDARD:
            t0 = time.perf_counter()
            generated_tokens = target_tokens[:requested_tokens]
            gen_time = (len(generated_tokens) * target_token_time) + target_profile.base_latency_ms
            cost = (len(generated_tokens) / 1_000_000.0) * target_profile.cost_per_1m_output_usd
            compute = len(generated_tokens) * target_profile.flops_per_token

            state.generated_tokens = len(generated_tokens)
            state.total_latency_ms = round(gen_time, 2)
            state.estimated_cost_usd = round(cost, 7)
            state.total_compute_flops = compute
            state.realized_speedup = 1.0
            state.realized_utility = self.economics.compute_utility(
                quality=target_profile.quality_score,
                latency_ms=gen_time,
                cost_usd=cost,
                compute_flops=compute,
            )

            telemetry = self._build_telemetry(state, selected_cand, ControllerAction.STANDARD_DECODE, DecisionReason.STANDARD_DECODE_PREFERRED)
            return " ".join(generated_tokens), state, telemetry

        # Speculative Decoding Execution Loop
        while len(generated_tokens) < requested_tokens and state.rounds_executed < self.config.max_speculation_rounds:
            state.rounds_executed += 1
            remaining = requested_tokens - len(generated_tokens)
            curr_draft_len = min(state.draft_length, remaining)

            # Check for acceptance deterioration fallback
            if self.estimator.is_deteriorating(window=3):
                state.fallback_occurred = True
                state.fallback_reason = "Acceptance rate deterioration detected across 3 consecutive rounds."
                state.current_strategy = SpeculationStrategy.STANDARD
                # Complete remainder via standard decoding
                rem_tokens = target_tokens[token_cursor : token_cursor + remaining]
                generated_tokens.extend(rem_tokens)
                token_cursor += len(rem_tokens)
                state.generated_tokens = len(generated_tokens)
                state.total_latency_ms += (len(rem_tokens) * target_token_time)
                state.estimated_cost_usd += (len(rem_tokens) / 1_000_000.0) * target_profile.cost_per_1m_output_usd
                state.total_compute_flops += len(rem_tokens) * target_profile.flops_per_token
                break

            # 2. Draft Proposal
            proposal = self._generate_draft_proposal(
                strategy=state.current_strategy,
                draft_model=state.draft_model,
                draft_length=curr_draft_len,
                current_text=" ".join(generated_tokens) if generated_tokens else full_prompt,
                target_reference=target_tokens[token_cursor:],
            )
            state.drafted_tokens += len(proposal.tokens)
            state.draft_latency_ms += proposal.draft_latency_ms
            state.draft_compute_flops += proposal.draft_compute_flops
            state.estimated_cost_usd += proposal.draft_cost_usd

            # 3. Verification
            v_res = self.verifier.verify_draft(
                target_profile=target_profile,
                target_reference_tokens=target_tokens[token_cursor:],
                proposal=proposal,
            )

            accepted_count = len(v_res.accepted_tokens)
            state.accepted_tokens += accepted_count
            state.rejected_tokens += len(v_res.rejected_tokens)
            state.verification_latency_ms += v_res.verification_latency_ms
            state.verification_compute_flops += v_res.verification_compute_flops
            state.estimated_cost_usd += v_res.verification_cost_usd

            # Advance generated tokens
            generated_tokens.extend(v_res.accepted_tokens)
            token_cursor += accepted_count
            if v_res.replacement_token and len(generated_tokens) < requested_tokens:
                generated_tokens.append(v_res.replacement_token)
                token_cursor += 1

            # 4. Online Acceptance Update
            round_rate = self.estimator.record_round(
                strategy=state.current_strategy,
                draft_model=state.draft_model,
                draft_length=curr_draft_len,
                drafted_count=len(proposal.tokens),
                accepted_count=accepted_count,
            )
            state.rolling_acceptance_history.append(round_rate)

            # 5. Adaptive Draft Length Adjustment
            new_len, len_reason = self.optimizer.adapt_length_on_feedback(
                current_length=state.draft_length,
                realized_acceptance=round_rate,
                break_even_rate=selected_cand.break_even_acceptance,
            )
            state.draft_length = new_len

            # Record round step
            state.history.append({
                "round": state.rounds_executed,
                "strategy": state.current_strategy.value,
                "draft_length": curr_draft_len,
                "drafted": len(proposal.tokens),
                "accepted": accepted_count,
                "round_acceptance": round_rate,
                "adaptation": len_reason,
            })

        # Calculate Final Totals
        state.generated_tokens = len(generated_tokens)
        state.total_latency_ms = round(target_profile.base_latency_ms + state.draft_latency_ms + state.verification_latency_ms, 2)
        state.total_compute_flops = state.draft_compute_flops + state.verification_compute_flops
        state.acceptance_rate = round(state.accepted_tokens / max(1, state.drafted_tokens), 4)
        state.average_accepted_tokens_per_round = round(state.accepted_tokens / max(1, state.rounds_executed), 2)
        
        # Realized speedup vs theoretical standard autoregressive time
        std_time = (state.generated_tokens * target_token_time) + target_profile.base_latency_ms
        state.realized_speedup = round(std_time / max(0.01, state.total_latency_ms), 2)
        state.realized_utility = round(
            self.economics.compute_utility(
                quality=target_profile.quality_score,
                latency_ms=state.total_latency_ms,
                cost_usd=state.estimated_cost_usd,
                compute_flops=state.total_compute_flops,
            ),
            4,
        )

        action = ControllerAction.FALLBACK_REROUTE if state.fallback_occurred else ControllerAction.SPECULATE_DRAFT_MODEL
        reason = DecisionReason.SPECULATION_DETERIORATED_FALLBACK if state.fallback_occurred else DecisionReason.SPECULATION_FASTER_EXPECTED

        telemetry = self._build_telemetry(state, selected_cand, action, reason)
        return " ".join(generated_tokens), state, telemetry

    def _generate_draft_proposal(
        self,
        strategy: SpeculationStrategy,
        draft_model: Optional[str],
        draft_length: int,
        current_text: str,
        target_reference: List[str],
    ) -> DraftTokenProposal:
        """Generate candidate draft tokens based on active strategy."""
        t0 = time.perf_counter()

        if strategy == SpeculationStrategy.NGRAM:
            prop = self.ngram_engine.speculate(current_text, draft_length=draft_length)
            if prop:
                return prop
            return DraftTokenProposal(
                tokens=[],
                confidences=[],
                strategy=SpeculationStrategy.NGRAM,
                draft_model="ngram-table",
                draft_length=0,
                draft_latency_ms=0.01,
                draft_cost_usd=0.0,
                draft_compute_flops=1000.0,
                metadata={"ngram_miss": True},
            )

        # Draft model proposal (simulated calibrated generation matching draft model profile)
        draft_profile = self.registry.get(draft_model) if draft_model else None
        if not draft_profile:
            draft_profile = self.registry.get("aeon-draft-0.5b") or self.registry.get("aeon-small-1.5b")

        # Simulate drafting from reference with draft-model error rate ~ (1 - quality)
        draft_tokens: List[str] = []
        confidences: List[float] = []
        error_probability = max(0.05, 1.0 - (draft_profile.quality_score if draft_profile else 0.80))

        for i in range(min(draft_length, len(target_reference))):
            ref_tok = target_reference[i]
            # Draft model accurately predicts with probability (1 - error_probability)
            if (i + 1) * 0.12 < (1.0 - error_probability):
                draft_tokens.append(ref_tok)
                confidences.append(0.85)
            else:
                # Slight variant or alternative token
                draft_tokens.append(ref_tok + "_alt" if not ref_tok.endswith(".") else ref_tok)
                confidences.append(0.55)

        if not draft_tokens and target_reference:
            draft_tokens = [target_reference[0]]
            confidences = [0.80]

        draft_time_per_tok = 1000.0 / max(1.0, draft_profile.tokens_per_second if draft_profile else 500.0)
        draft_latency = len(draft_tokens) * draft_time_per_tok
        draft_cost = (len(draft_tokens) / 1_000_000.0) * (draft_profile.cost_per_1m_output_usd if draft_profile else 0.05)
        draft_flops = len(draft_tokens) * (draft_profile.flops_per_token if draft_profile else 1e9)

        return DraftTokenProposal(
            tokens=draft_tokens,
            confidences=confidences,
            strategy=strategy,
            draft_model=draft_profile.name if draft_profile else "draft-sim",
            draft_length=len(draft_tokens),
            draft_latency_ms=round(draft_latency, 2),
            draft_cost_usd=round(draft_cost, 7),
            draft_compute_flops=draft_flops,
        )

    def _synthesize_target_sequence(
        self,
        prompt: str,
        count: int,
        reference_text: Optional[str] = None,
    ) -> List[str]:
        """Produce predictable ground-truth reference sequence for target model."""
        if reference_text:
            tokens = reference_text.strip().split()
            if len(tokens) >= count:
                return tokens[:count]
            # Pad if needed
            return (tokens * ((count // len(tokens)) + 1))[:count]

        # Natural sequence synthesis
        base_words = [
            "the", "optimal", "inference", "path", "minimizes", "latency", "and", "computational",
            "waste", "while", "preserving", "exact", "task", "fidelity", "across", "heterogeneous",
            "models", "and", "deterministic", "tools", "in", "the", "system", "architecture",
        ]
        return (base_words * ((count // len(base_words)) + 1))[:count]

    def _build_telemetry(
        self,
        state: SpeculationState,
        candidate: SpeculationCandidate,
        action: ControllerAction,
        reason: DecisionReason,
    ) -> Telemetry:
        """Construct structured Telemetry conforming to Phase 5 Specification Section 19."""
        return Telemetry(
            request_id=state.request_id,
            task_type="generation",
            route="speculative" if state.current_strategy != SpeculationStrategy.STANDARD else "standard",
            model=state.target_model,
            reasoning_mode="none",
            input_tokens=15,
            output_tokens=state.generated_tokens,
            reasoning_tokens=0,
            latency_ms=state.total_latency_ms,
            compute_estimate=state.total_compute_flops,
            memory_mb=state.memory_usage_mb + 10.0,
            estimated_energy=round((state.total_latency_ms / 1000.0) * 80.0, 4),
            estimated_cost=state.estimated_cost_usd,
            cache_hit=False,
            speculative=state.current_strategy != SpeculationStrategy.STANDARD,
            verification=True,
            confidence=state.confidence,
            escalated=False,
            correct=True,
            decision_reason=reason.value,
            stopping_reason=reason.value,
            # Phase 5 Fields
            speculation_strategy=state.current_strategy.value,
            speculation_action=action.value,
            draft_model=state.draft_model,
            draft_length=state.draft_length,
            drafted_tokens=state.drafted_tokens,
            accepted_tokens=state.accepted_tokens,
            rejected_tokens=state.rejected_tokens,
            acceptance_rate=state.acceptance_rate,
            average_accepted_per_round=state.average_accepted_tokens_per_round,
            draft_latency_ms=round(state.draft_latency_ms, 2),
            verification_latency_ms=round(state.verification_latency_ms, 2),
            speculation_compute_flops=state.total_compute_flops,
            expected_speedup=candidate.expected_speedup,
            realized_speedup=state.realized_speedup,
            expected_utility=candidate.expected_utility,
            realized_utility=state.realized_utility,
            break_even_acceptance=candidate.break_even_acceptance,
            speculation_mode=state.execution_mode,
            speculation_rounds=state.rounds_executed,
            fallback_occurred=state.fallback_occurred,
        )


# Global controller instance
global_speculation_controller = SpeculationController()
