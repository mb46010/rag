"""Configuration for retrieval system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalConfig:
    """Configuration for retrieval behavior."""

    # Connection settings
    weaviate_url: str = "http://localhost:8080"
    collection_name: str = "PolicyChunk"
    embedding_model: str = "text-embedding-3-large"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval settings
    default_top_k: int = 5
    candidate_pool_multiplier: int = 4  # Fetch 4x candidates for reranking
    min_confidence: float = 0.3

    # Adaptive alpha settings (BM25 vs Vector balance)
    # Lower alpha = more BM25 (keyword), Higher alpha = more vector (semantic)
    alpha_factual: float = 0.3  # Specific facts benefit from keyword matching
    alpha_conceptual: float = 0.7  # Explanations benefit from semantic search
    alpha_procedural: float = 0.5  # How-to questions need balance
    alpha_default: float = 0.5

    # Confidence thresholds for calibrated reranking
    high_confidence_threshold: float = 0.7
    medium_confidence_threshold: float = 0.4

    # Feature flags
    enable_reranking: bool = True
    enable_context_window: bool = True
    enable_rrf: bool = False  # Use RRF instead of Weaviate's fusion
    enable_hyde: bool = False  # Hypothetical Document Embeddings

    # RRF parameter
    rrf_k: int = 60  # Standard RRF constant
