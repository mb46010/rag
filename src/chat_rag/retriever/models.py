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
    section_path: List[str]
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
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # Enhanced metadata
    confidence_level: str = "unknown"
    rerank_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None

    # For debugging/observability
    query_type: Optional[str] = None
    alpha_used: Optional[float] = None

    # HyDE metadata
    hyde_enhanced: bool = False
    hypothetical_used: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "section_path_str": self.section_path_str,
            "section_path": self.section_path,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "topic": self.topic,
            "country": self.country,
            "score": self.score,
            "previous_chunk": self.previous_chunk,
            "next_chunk": self.next_chunk,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "confidence_level": self.confidence_level,
            "rerank_score": self.rerank_score,
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
            "query_type": self.query_type,
            "alpha_used": self.alpha_used,
            "hyde_enhanced": self.hyde_enhanced,
            "hypothetical_used": self.hypothetical_used,
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
