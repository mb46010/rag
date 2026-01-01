"""Data models for retrieval system."""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class QueryType(Enum):
    """Query classification for adaptive retrieval."""

    FACTUAL = "factual"  # Specific facts, numbers, dates
    CONCEPTUAL = "conceptual"  # Explanations, processes
    PROCEDURAL = "procedural"  # How-to, steps
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for retrieved chunks."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    LOW_BUT_BEST = "low_but_best"  # Low confidence but best available


@dataclass
class PolicyChunk:
    """Retrieved policy chunk with metadata and context."""

    text: str
    document_id: str
    document_name: str
    section_path_str: str
    chunk_id: str
    chunk_index: int
    topic: str
    country: str
    active: bool
    last_modified: str
    score: float = 0.0

    # Context window
    previous_chunk: Optional[str] = None
    next_chunk: Optional[str] = None

    # Enhanced metadata
    confidence_level: str = "unknown"
    rerank_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None

    # For debugging/observability
    query_type: Optional[str] = None
    alpha_used: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "section_path_str": self.section_path_str,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "topic": self.topic,
            "country": self.country,
            "score": self.score,
            "confidence_level": self.confidence_level,
        }


@dataclass
class RetrievalResult:
    """Result of a retrieval operation with metadata."""

    chunks: List[PolicyChunk]
    query: str
    query_type: QueryType
    alpha_used: float
    total_candidates: int
    filters_applied: dict = field(default_factory=dict)

    @property
    def has_high_confidence(self) -> bool:
        """Check if any chunk has high confidence."""
        return any(c.confidence_level == "high" for c in self.chunks)

    @property
    def top_score(self) -> float:
        """Get the top chunk score."""
        return self.chunks[0].score if self.chunks else 0.0
