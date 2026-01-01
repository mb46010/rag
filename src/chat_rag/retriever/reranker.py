"""
Reranker module with calibrated scoring.

Uses CrossEncoder from sentence-transformers for semantic reranking
of retrieved chunks.
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import CrossEncoder
from langfuse import observe

from .models import PolicyChunk

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranker using CrossEncoder with score calibration.

    The calibration converts raw CrossEncoder scores (which can be
    arbitrary real numbers) into interpretable confidence scores
    between 0 and 1 using sigmoid transformation.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        """
        Initialize the reranker.

        Args:
            model_name: Name of the CrossEncoder model from HuggingFace
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        logger.info(f"Initializing Reranker with model: {model_name}")
        self.model = CrossEncoder(model_name, device=device)

        # Calibration thresholds (can be tuned with validation data)
        self.high_confidence_threshold = 0.7
        self.medium_confidence_threshold = 0.4

    @observe(as_type="generation")
    def rerank(
        self,
        query: str,
        chunks: List[PolicyChunk],
        top_k: Optional[int] = None,
        return_scores: bool = False,
    ) -> List[PolicyChunk]:
        """
        Rerank a list of chunks based on the query.

        Args:
            query: The search query
            chunks: List of PolicyChunk objects to rerank
            top_k: Number of top chunks to return (None = return all)
            return_scores: If True, also return raw scores

        Returns:
            Reranked list of PolicyChunk objects with updated scores
        """
        if not chunks:
            return []

        logger.info(f"Reranking {len(chunks)} chunks for query: '{query[:50]}...'")

        # Prepare query-document pairs
        pairs = [[query, chunk.text] for chunk in chunks]

        # Get raw scores from CrossEncoder
        raw_scores = self.model.predict(pairs)

        # Apply sigmoid calibration for interpretable confidence scores
        calibrated_scores = self._calibrate_scores(raw_scores)

        # Assign scores and confidence levels to chunks
        for chunk, raw, calibrated in zip(chunks, raw_scores, calibrated_scores):
            chunk.rerank_score = float(calibrated)
            chunk.score = float(calibrated)
            chunk.confidence_level = self._get_confidence_level(calibrated)

        # Sort by calibrated score descending
        chunks.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            chunks = chunks[:top_k]

        return chunks

    def _calibrate_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Calibrate raw CrossEncoder scores using sigmoid.

        This transforms arbitrary real-valued scores into
        probabilities between 0 and 1.
        """
        return 1 / (1 + np.exp(-raw_scores))

    def _get_confidence_level(self, score: float) -> str:
        """Map score to human-readable confidence level."""
        if score >= self.high_confidence_threshold:
            return "high"
        elif score >= self.medium_confidence_threshold:
            return "medium"
        return "low"

    @observe(as_type="generation")
    def score_single(self, query: str, text: str) -> float:
        """
        Score a single query-text pair.

        Useful for ad-hoc relevance checking.
        """
        raw_score = self.model.predict([[query, text]])[0]
        return float(1 / (1 + np.exp(-raw_score)))
