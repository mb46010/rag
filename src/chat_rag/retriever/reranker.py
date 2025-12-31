import logging
from typing import List, Optional
from sentence_transformers import CrossEncoder
from .models import PolicyChunk

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranker using CrossEncoder from sentence-transformers.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.

        Args:
            model_name: Name of the CrossEncoder model.
        """
        self.model_name = model_name
        logger.info(f"Initializing Reranker with model: {model_name}")
        # Use cpu device since we saw uv install torch cpu
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, chunks: List[PolicyChunk], top_k: Optional[int] = None) -> List[PolicyChunk]:
        """
        Rerank a list of chunks based on the query.

        Args:
            query: The search query.
            chunks: List of PolicyChunk objects to rerank.
            top_k: Number of top chunks to return. If None, returns all.

        Returns:
            Reranked list of PolicyChunk objects.
        """
        if not chunks:
            return []

        logger.info(f"Reranking {len(chunks)} chunks for query: '{query}'")

        # Prepare pairs for cross-encoder
        pairs = [[query, chunk.text] for chunk in chunks]

        # Predict scores
        scores = self.model.predict(pairs)

        # Assign scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        # Sort by score descending
        chunks.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            chunks = chunks[:top_k]

        return chunks
