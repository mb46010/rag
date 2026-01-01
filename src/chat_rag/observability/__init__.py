"""
Observability module for HR Assistant.

Provides cost tracking, latency monitoring, and metrics collection.
"""

from .cost_tracking import (
    CostTracker,
    QueryMetrics,
    TokenUsage,
    LatencyBreakdown,
    get_tracker,
    track_query,
    MODEL_PRICING,
)

__all__ = [
    "CostTracker",
    "QueryMetrics",
    "TokenUsage",
    "LatencyBreakdown",
    "get_tracker",
    "track_query",
    "MODEL_PRICING",
]
