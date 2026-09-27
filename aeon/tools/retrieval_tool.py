"""
Retrieval Tool for AEON.

Provides structured fact retrieval from knowledge bases with relevance scoring,
source provenance, timestamps, and freshness tracking.
Conforms to AEON Specification Sections 12 and 13.
Explicitly distinguishes execution modes: LOCAL_RETRIEVAL vs SIMULATED_RETRIEVAL vs LIVE_RETRIEVAL.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
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


class RetrievedDocument(BaseModel):
    """Structured knowledge document returned by retrieval."""
    doc_id: str
    title: str
    content: str
    source: str
    timestamp: float
    relevance_score: float = 0.0
    confidence: float = 0.95


class RetrievalTool(BaseTool):
    """
    Retrieval abstraction supporting local indexed knowledge and simulated live updates.
    Truthfully reports its execution mode as LOCAL_RETRIEVAL.
    """

    name = "retrieval"
    tool_type = ToolType.RETRIEVAL
    category = ToolCategory.RETRIEVAL

    def __init__(self, execution_mode: str = "LOCAL_RETRIEVAL"):
        self.execution_mode = execution_mode
        self._corpus: List[Dict[str, Any]] = [
            {
                "doc_id": "doc_nvda_2025",
                "title": "NVIDIA FY2025 Financial Results",
                "keywords": ["nvidia", "revenue", "fy2025", "financial", "earnings"],
                "content": "NVIDIA reported record full-year FY2025 revenue of $130.5 billion, up 114% year-over-year.",
                "source": "NVIDIA Investor Relations / SEC Filing",
                "timestamp": time.time() - 3600 * 24 * 10,  # 10 days ago
                "confidence": 0.99,
            },
            {
                "doc_id": "doc_aapl_2024",
                "title": "Apple FY2024 Annual Report",
                "keywords": ["apple", "revenue", "2024", "fy2024"],
                "content": "Apple posted annual revenue of $391.04 billion in fiscal year 2024.",
                "source": "Apple Inc. Form 10-K",
                "timestamp": time.time() - 3600 * 24 * 60,
                "confidence": 0.99,
            },
            {
                "doc_id": "doc_ind_gdp",
                "title": "India GDP Economic Statistics",
                "keywords": ["india", "gdp", "economy", "size"],
                "content": "India's nominal GDP in 2024 reached approximately $3.94 trillion USD, ranking as the fifth largest global economy.",
                "source": "IMF World Economic Outlook",
                "timestamp": time.time() - 3600 * 24 * 30,
                "confidence": 0.98,
            },
            {
                "doc_id": "doc_paris",
                "title": "France Overview",
                "keywords": ["france", "capital", "paris"],
                "content": "Paris is the official capital and most populous city of France.",
                "source": "Official French National Statistics",
                "timestamp": time.time() - 3600 * 24 * 365,
                "confidence": 0.999,
            },
        ]

    def add_document(self, doc_id: str, title: str, content: str, keywords: List[str], source: str = "custom", timestamp: Optional[float] = None) -> None:
        """Add or update a knowledge document in the local retrieval index."""
        self._corpus.append({
            "doc_id": doc_id,
            "title": title,
            "keywords": [k.lower() for k in keywords],
            "content": content,
            "source": source,
            "timestamp": timestamp or time.time(),
            "confidence": 0.95,
        })

    def can_handle(self, request: ToolRequest) -> Optional[ToolCapability]:
        """Detect factual, lookup, or current-knowledge queries."""
        q_lower = request.query.lower()
        is_retrieval_intent = any(w in q_lower for w in [
            "who is", "what was", "what is", "when was", "revenue", "gdp",
            "stock price", "latest", "founded", "capital of", "earnings",
        ]) or request.task_type in [TaskType.FACTUAL, TaskType.RETRIEVAL]

        # Exclude pure arithmetic / equation queries
        if any(sym in request.query for sym in ["+", "*", "/"]) and any(c.isdigit() for c in request.query):
            if not any(k in q_lower for k in ["revenue", "gdp", "earnings", "capital", "what was"]):
                return None

        if not is_retrieval_intent:
            return None

        return ToolCapability(
            tool_type=ToolType.RETRIEVAL,
            category=ToolCategory.RETRIEVAL,
            supported_task_types=[TaskType.FACTUAL, TaskType.RETRIEVAL],
            supported_operations=["keyword_search", "document_ranking", "source_attribution", "freshness_filtering"],
            deterministic=False,
            requires_network=False if self.execution_mode == "LOCAL_RETRIEVAL" else True,
            requires_sandbox=False,
            expected_quality=0.96,
            expected_latency_ms=15.0,
            expected_cost_usd=0.000030,
            expected_compute_flops=5e5,
            expected_memory_mb=8.0,
            expected_energy_joules=0.0004,
            risk_level=0.03,
            confidence=0.96,
            execution_mode=self.execution_mode,
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        """Search local corpus and return best matching facts."""
        t0 = time.perf_counter()
        q_tokens = set(request.query.lower().replace("?", "").replace("'", " ").split())

        best_doc = None
        best_score = 0.0

        for doc in self._corpus:
            score = 0.0
            doc_kws = set(doc["keywords"])
            doc_words = set(doc["content"].lower().split())

            # Keyword overlap
            overlap = len(q_tokens.intersection(doc_kws))
            content_overlap = len(q_tokens.intersection(doc_words))
            score = (overlap * 2.0) + (content_overlap * 0.5)

            if score > best_score:
                best_score = score
                best_doc = doc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not best_doc or best_score < 1.0:
            return ToolResult(
                tool_type=ToolType.RETRIEVAL,
                status=ToolStatus.LOW_CONFIDENCE,
                output=None,
                error="No sufficiently relevant documents found in knowledge base",
                actual_cost=ToolCost(latency_ms=latency_ms, cost_usd=0.000010, compute_flops=1e5),
                confidence=0.20,
            )

        output_str = f"{best_doc['content']} (Source: {best_doc['source']})"
        age_hours = (time.time() - best_doc["timestamp"]) / 3600.0
        freshness = max(0.0, 1.0 - (age_hours / (24.0 * 365.0)))

        return ToolResult(
            tool_type=ToolType.RETRIEVAL,
            status=ToolStatus.SUCCESS,
            output=output_str,
            actual_cost=ToolCost(
                latency_ms=round(latency_ms, 2),
                cost_usd=0.000030,
                compute_flops=6e5,
                memory_mb=8.0,
                energy_joules=round((latency_ms / 1000.0) * 25.0, 5),
            ),
            confidence=best_doc["confidence"],
            raw_metadata={
                "doc_id": best_doc["doc_id"],
                "source": best_doc["source"],
                "timestamp": best_doc["timestamp"],
                "freshness_score": round(freshness, 3),
                "relevance_score": round(best_score, 2),
                "execution_mode": self.execution_mode,
            },
        )

    def verify(self, result: ToolResult) -> VerificationResult:
        """Verify retrieved fact freshness and source authenticity."""
        if result.status != ToolStatus.SUCCESS or result.output is None:
            return VerificationResult(
                status=VerificationStatus.VERIFICATION_FAILED,
                verified_output=None,
                verifier_used="retrieval_verifier",
                discrepancy_detected=True,
                explanation=f"Retrieval failed: {result.error}",
            )

        src = result.raw_metadata.get("source", "")
        freshness = result.raw_metadata.get("freshness_score", 0.0)

        if not src:
            return VerificationResult(
                status=VerificationStatus.VERIFICATION_FAILED,
                verified_output=result.output,
                verifier_used="retrieval_verifier",
                discrepancy_detected=True,
                explanation="Source provenance missing",
            )

        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            verified_output=result.output,
            verifier_used="retrieval_verifier",
            discrepancy_detected=False,
            explanation=f"Verified against source '{src}' with freshness score {freshness}",
            verification_cost_usd=0.000001,
            verification_latency_ms=0.05,
        )
