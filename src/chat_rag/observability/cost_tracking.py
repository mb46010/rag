"""
Cost Tracking Module

Tracks per-query costs including:
- LLM token usage (input/output)
- Embedding tokens
- Latency breakdown by component
- Aggregated metrics for dashboards

Integrates with Langfuse for production observability.
"""

import logging
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (as of 2024, update as needed)
MODEL_PRICING = {
    # GPT-4o
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # GPT-4
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    # GPT-3.5
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Embeddings
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
}


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def cost_usd(self) -> float:
        """Calculate cost in USD."""
        pricing = MODEL_PRICING.get(self.model, {"input": 0, "output": 0})
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class LatencyBreakdown:
    """Latency tracking for query components."""
    total_ms: float = 0
    intent_classification_ms: float = 0
    retrieval_ms: float = 0
    reranking_ms: float = 0
    mcp_tools_ms: float = 0
    synthesis_ms: float = 0
    hyde_ms: float = 0
    embedding_ms: float = 0


@dataclass
class QueryMetrics:
    """Complete metrics for a single query."""
    query_id: str
    timestamp: datetime
    user_email: str
    query_text: str
    
    # Token usage
    token_usage: List[TokenUsage] = field(default_factory=list)
    
    # Latency
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    
    # Query characteristics
    intent: Optional[str] = None
    query_type: Optional[str] = None
    country_filter: Optional[str] = None
    
    # Retrieval metrics
    chunks_retrieved: int = 0
    top_chunk_score: float = 0
    hyde_used: bool = False
    
    # Outcome
    success: bool = True
    error: Optional[str] = None
    clarification_requested: bool = False
    
    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.token_usage)
    
    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.token_usage)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage."""
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat(),
            "user_email": self.user_email,
            "query_text": self.query_text[:100],
            "intent": self.intent,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": self.latency.total_ms,
            "chunks_retrieved": self.chunks_retrieved,
            "success": self.success,
            "hyde_used": self.hyde_used,
        }


class CostTracker:
    """
    Tracks costs and latencies for queries.
    
    Usage:
        tracker = CostTracker()
        
        with tracker.track_query("q123", "user@co.com", "How many PTO days?") as metrics:
            with tracker.track_latency(metrics, "retrieval"):
                chunks = retriever.retrieve(query)
            
            tracker.add_token_usage(metrics, "gpt-4o", input=500, output=200)
        
        # Get aggregated stats
        print(tracker.get_summary())
    """
    
    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.queries: List[QueryMetrics] = []
        self._current_query: Optional[QueryMetrics] = None
    
    @contextmanager
    def track_query(
        self,
        query_id: str,
        user_email: str,
        query_text: str,
    ):
        """Context manager for tracking a complete query."""
        import uuid
        
        metrics = QueryMetrics(
            query_id=query_id or str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            user_email=user_email,
            query_text=query_text,
        )
        
        self._current_query = metrics
        start_time = time.perf_counter()
        
        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            raise
        finally:
            metrics.latency.total_ms = (time.perf_counter() - start_time) * 1000
            self.queries.append(metrics)
            self._current_query = None
            
            if self.enable_logging:
                self._log_metrics(metrics)
    
    @contextmanager
    def track_latency(self, metrics: QueryMetrics, component: str):
        """Track latency for a specific component."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Set the appropriate latency field
            latency_fields = {
                "intent_classification": "intent_classification_ms",
                "retrieval": "retrieval_ms",
                "reranking": "reranking_ms",
                "mcp_tools": "mcp_tools_ms",
                "synthesis": "synthesis_ms",
                "hyde": "hyde_ms",
                "embedding": "embedding_ms",
            }
            
            if component in latency_fields:
                setattr(metrics.latency, latency_fields[component], elapsed_ms)
    
    def add_token_usage(
        self,
        metrics: QueryMetrics,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Add token usage for an LLM call."""
        usage = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        metrics.token_usage.append(usage)
    
    def _log_metrics(self, metrics: QueryMetrics):
        """Log metrics for observability."""
        log_data = metrics.to_dict()
        logger.info(f"Query metrics: {json.dumps(log_data)}")
    
    def get_summary(self, last_n: int = None) -> Dict[str, Any]:
        """Get aggregated summary of tracked queries."""
        queries = self.queries[-last_n:] if last_n else self.queries
        
        if not queries:
            return {"total_queries": 0}
        
        total_cost = sum(q.total_cost_usd for q in queries)
        total_tokens = sum(q.total_tokens for q in queries)
        avg_latency = sum(q.latency.total_ms for q in queries) / len(queries)
        success_rate = sum(1 for q in queries if q.success) / len(queries)
        
        # Cost by model
        cost_by_model = defaultdict(float)
        for q in queries:
            for usage in q.token_usage:
                cost_by_model[usage.model] += usage.cost_usd
        
        # Latency breakdown
        avg_latency_breakdown = {
            "retrieval_ms": sum(q.latency.retrieval_ms for q in queries) / len(queries),
            "synthesis_ms": sum(q.latency.synthesis_ms for q in queries) / len(queries),
            "mcp_tools_ms": sum(q.latency.mcp_tools_ms for q in queries) / len(queries),
        }
        
        return {
            "total_queries": len(queries),
            "total_cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
            "avg_cost_per_query_usd": round(total_cost / len(queries), 6),
            "avg_tokens_per_query": total_tokens // len(queries),
            "avg_latency_ms": round(avg_latency, 1),
            "success_rate": round(success_rate, 3),
            "cost_by_model": dict(cost_by_model),
            "avg_latency_breakdown": avg_latency_breakdown,
            "hyde_usage_rate": sum(1 for q in queries if q.hyde_used) / len(queries),
        }
    
    def get_cost_report(self, period_hours: int = 24) -> Dict[str, Any]:
        """Get cost report for a time period."""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(hours=period_hours)
        recent = [q for q in self.queries if q.timestamp >= cutoff]
        
        if not recent:
            return {"period_hours": period_hours, "total_queries": 0}
        
        summary = self.get_summary()
        summary["period_hours"] = period_hours
        
        # Add hourly breakdown
        hourly_costs = defaultdict(float)
        for q in recent:
            hour_key = q.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_costs[hour_key] += q.total_cost_usd
        
        summary["hourly_costs"] = dict(hourly_costs)
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "queries": [q.to_dict() for q in self.queries],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.queries)} query metrics to {filepath}")


# Global tracker instance
_global_tracker: Optional[CostTracker] = None


def get_tracker() -> CostTracker:
    """Get or create global tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def track_query(query_id: str, user_email: str, query_text: str):
    """Convenience function for tracking queries."""
    return get_tracker().track_query(query_id, user_email, query_text)
