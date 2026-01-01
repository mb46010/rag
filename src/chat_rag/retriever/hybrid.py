"""
Enhanced Hybrid Retriever with:
- Query-adaptive alpha
- Calibrated reranking
- RRF fusion option
- Confidence scoring
- Batch context fetching
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import weaviate
from weaviate.classes.query import MetadataQuery, Filter
from llama_index.embeddings.openai import OpenAIEmbedding
from sentence_transformers import CrossEncoder

from .models import PolicyChunk, QueryType, RetrievalResult
from .config import RetrievalConfig

logger = logging.getLogger(__name__)


class EnhancedHybridRetriever:
    """
    Production-grade hybrid retriever with adaptive behavior.

    Features:
    - Query classification for optimal BM25/vector balance
    - CrossEncoder reranking with calibrated confidence scores
    - Optional Reciprocal Rank Fusion (RRF) for stable ranking
    - Batch fetching of context windows
    - Confidence-based filtering
    """

    def __init__(self, config: RetrievalConfig = None):
        self.config = config or RetrievalConfig()
        logger.info("Initializing EnhancedHybridRetriever")

        # Initialize Weaviate client
        host = self.config.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0]
        self.client = weaviate.connect_to_local(host=host)

        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(model=self.config.embedding_model)
        logger.info(f"Using embedding model: {self.config.embedding_model}")

        # Initialize reranker
        if self.config.enable_reranking:
            logger.info(f"Loading reranker: {self.config.reranker_model}")
            self.reranker = CrossEncoder(self.config.reranker_model, device="cpu")
        else:
            self.reranker = None

        # Initialize query classification patterns
        self._init_query_patterns()

    def _init_query_patterns(self):
        """Initialize keyword patterns for query classification."""
        self.factual_keywords = {
            "how many",
            "what is the",
            "how much",
            "when",
            "where",
            "limit",
            "maximum",
            "minimum",
            "deadline",
            "amount",
            "number of",
            "rate",
            "percentage",
            "days",
            "hours",
        }
        self.procedural_keywords = {
            "how do i",
            "how to",
            "steps",
            "process",
            "procedure",
            "submit",
            "request",
            "apply",
            "file",
            "register",
            "enroll",
            "claim",
        }

    def classify_query(self, query: str) -> QueryType:
        """
        Classify query type for adaptive alpha selection.

        Args:
            query: The search query

        Returns:
            QueryType enum value
        """
        query_lower = query.lower()

        if any(kw in query_lower for kw in self.factual_keywords):
            return QueryType.FACTUAL
        elif any(kw in query_lower for kw in self.procedural_keywords):
            return QueryType.PROCEDURAL
        else:
            return QueryType.CONCEPTUAL

    def get_adaptive_alpha(self, query_type: QueryType) -> float:
        """
        Get optimal alpha based on query type.

        Args:
            query_type: Classified query type

        Returns:
            Alpha value (0=BM25, 1=vector, 0.5=balanced)
        """
        alpha_map = {
            QueryType.FACTUAL: self.config.alpha_factual,
            QueryType.CONCEPTUAL: self.config.alpha_conceptual,
            QueryType.PROCEDURAL: self.config.alpha_procedural,
            QueryType.UNKNOWN: self.config.alpha_default,
        }
        return alpha_map.get(query_type, self.config.alpha_default)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        override_alpha: Optional[float] = None,
    ) -> List[PolicyChunk]:
        """
        Retrieve relevant policy chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"country": "CH"})
            override_alpha: Override adaptive alpha selection

        Returns:
            List of PolicyChunk objects with confidence scores
        """
        top_k = top_k or self.config.default_top_k

        # Ensure active filter is always applied
        filters = filters.copy() if filters else {}
        filters.setdefault("active", True)

        # Classify query and determine alpha
        query_type = self.classify_query(query)
        alpha = override_alpha if override_alpha is not None else self.get_adaptive_alpha(query_type)

        logger.info(f"Query: '{query[:50]}...' | Type: {query_type.value} | Alpha: {alpha}")

        # Calculate candidate pool size
        candidate_size = top_k * self.config.candidate_pool_multiplier

        # Retrieve candidates
        if self.config.enable_rrf:
            chunks = self._retrieve_with_rrf(query, candidate_size, filters)
        else:
            chunks = self._retrieve_hybrid(query, candidate_size, filters, alpha)

        if not chunks:
            logger.warning("No results found")
            return []

        logger.info(f"Retrieved {len(chunks)} candidates")

        # Rerank if enabled
        if self.config.enable_reranking and self.reranker:
            chunks = self._rerank_chunks(query, chunks, top_k)
        else:
            chunks = chunks[:top_k]

        # Assign confidence levels
        self._assign_confidence_levels(chunks)

        # Fetch context windows if enabled
        if self.config.enable_context_window:
            self._batch_fetch_context(chunks)

        # Add metadata for observability
        for chunk in chunks:
            chunk.query_type = query_type.value
            chunk.alpha_used = alpha

        # Filter low confidence (but keep at least one result)
        confident_chunks = [c for c in chunks if c.confidence_level != "low"]
        if not confident_chunks and chunks:
            chunks[0].confidence_level = "low_but_best"
            return [chunks[0]]

        return confident_chunks

    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve with full metadata for observability.

        Returns:
            RetrievalResult with chunks and retrieval metadata
        """
        top_k = top_k or self.config.default_top_k
        filters = filters.copy() if filters else {}
        filters.setdefault("active", True)

        query_type = self.classify_query(query)
        alpha = self.get_adaptive_alpha(query_type)

        candidate_size = top_k * self.config.candidate_pool_multiplier
        chunks = self._retrieve_hybrid(query, candidate_size, filters, alpha)
        total_candidates = len(chunks)

        if self.config.enable_reranking and self.reranker:
            chunks = self._rerank_chunks(query, chunks, top_k)
        else:
            chunks = chunks[:top_k]

        self._assign_confidence_levels(chunks)

        if self.config.enable_context_window:
            self._batch_fetch_context(chunks)

        return RetrievalResult(
            chunks=chunks,
            query=query,
            query_type=query_type,
            alpha_used=alpha,
            total_candidates=total_candidates,
            filters_applied=filters,
        )

    def _retrieve_hybrid(
        self, query: str, limit: int, filters: Dict[str, Any], alpha: float
    ) -> List[PolicyChunk]:
        """Standard Weaviate hybrid search."""
        collection = self.client.collections.get(self.config.collection_name)
        weaviate_filters = self._build_filters(filters)

        # Generate query embedding
        query_embedding = self.embed_model.get_query_embedding(query)

        response = collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            limit=limit,
            filters=weaviate_filters,
            return_metadata=MetadataQuery(score=True),
        )

        return [self._to_policy_chunk(obj) for obj in response.objects]

    def _retrieve_with_rrf(
        self, query: str, limit: int, filters: Dict[str, Any]
    ) -> List[PolicyChunk]:
        """
        Reciprocal Rank Fusion: run BM25 and vector separately, then fuse.

        RRF provides more stable ranking than score-based fusion because it
        only considers rank positions, not raw scores which have different
        distributions.
        """
        collection = self.client.collections.get(self.config.collection_name)
        weaviate_filters = self._build_filters(filters)
        query_embedding = self.embed_model.get_query_embedding(query)

        k = self.config.rrf_k

        # BM25 search
        bm25_response = collection.query.bm25(
            query=query,
            limit=limit,
            filters=weaviate_filters,
        )

        # Vector search
        vector_response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            filters=weaviate_filters,
        )

        # Build RRF scores
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, PolicyChunk] = {}

        for rank, obj in enumerate(bm25_response.objects):
            chunk_id = obj.properties.get("chunk_id")
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            chunk = self._to_policy_chunk(obj)
            chunk.bm25_rank = rank
            chunk_map[chunk_id] = chunk

        for rank, obj in enumerate(vector_response.objects):
            chunk_id = obj.properties.get("chunk_id")
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            if chunk_id not in chunk_map:
                chunk = self._to_policy_chunk(obj)
                chunk_map[chunk_id] = chunk
            chunk_map[chunk_id].vector_rank = rank

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        result = []
        for chunk_id in sorted_ids[:limit]:
            chunk = chunk_map[chunk_id]
            chunk.score = scores[chunk_id]
            result.append(chunk)

        return result

    def _rerank_chunks(
        self, query: str, chunks: List[PolicyChunk], top_k: int
    ) -> List[PolicyChunk]:
        """
        Rerank using CrossEncoder with calibrated scores.

        Uses sigmoid calibration to convert raw CrossEncoder scores
        to interpretable confidence scores between 0 and 1.
        """
        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]
        raw_scores = self.reranker.predict(pairs)

        # Sigmoid calibration for interpretable scores
        calibrated = 1 / (1 + np.exp(-raw_scores))

        for chunk, raw, cal in zip(chunks, raw_scores, calibrated):
            chunk.rerank_score = float(cal)
            chunk.score = float(cal)  # Override hybrid score with calibrated score

        chunks.sort(key=lambda x: x.score, reverse=True)
        return chunks[:top_k]

    def _assign_confidence_levels(self, chunks: List[PolicyChunk]):
        """Assign human-readable confidence levels based on scores."""
        for chunk in chunks:
            if chunk.score >= self.config.high_confidence_threshold:
                chunk.confidence_level = "high"
            elif chunk.score >= self.config.medium_confidence_threshold:
                chunk.confidence_level = "medium"
            else:
                chunk.confidence_level = "low"

    def _batch_fetch_context(self, chunks: List[PolicyChunk]):
        """
        Fetch adjacent chunks in a single batch query.

        This avoids the N+1 query problem where each chunk triggers
        separate queries for previous/next context.
        """
        if not chunks:
            return

        # Build list of adjacent indices to fetch
        adjacent_requests = []
        for chunk in chunks:
            if chunk.chunk_index > 0:
                adjacent_requests.append((chunk.document_id, chunk.chunk_index - 1))
            adjacent_requests.append((chunk.document_id, chunk.chunk_index + 1))

        if not adjacent_requests:
            return

        # Build OR filter for all adjacent chunks
        collection = self.client.collections.get(self.config.collection_name)

        filter_expr = None
        for doc_id, idx in adjacent_requests:
            f = Filter.by_property("document_id").equal(doc_id) & Filter.by_property("chunk_index").equal(idx)
            filter_expr = f if filter_expr is None else (filter_expr | f)

        try:
            response = collection.query.fetch_objects(filters=filter_expr, limit=len(adjacent_requests))

            # Build lookup map
            context_map = {}
            for obj in response.objects:
                key = (obj.properties["document_id"], obj.properties["chunk_index"])
                context_map[key] = obj.properties["text"]

            # Assign to chunks
            for chunk in chunks:
                chunk.previous_chunk = context_map.get((chunk.document_id, chunk.chunk_index - 1))
                chunk.next_chunk = context_map.get((chunk.document_id, chunk.chunk_index + 1))

        except Exception as e:
            logger.warning(f"Batch context fetch failed: {e}")

    def _build_filters(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """Build Weaviate filter from dictionary."""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            conditions.append(Filter.by_property(key).equal(value))

        result = conditions[0]
        for condition in conditions[1:]:
            result = result & condition

        return result

    def _to_policy_chunk(self, obj) -> PolicyChunk:
        """Convert Weaviate object to PolicyChunk."""
        props = obj.properties
        return PolicyChunk(
            text=props.get("text", ""),
            document_id=props.get("document_id", ""),
            document_name=props.get("document_name", ""),
            section_path_str=props.get("section_path_str", ""),
            chunk_id=props.get("chunk_id", ""),
            chunk_index=props.get("chunk_index", 0),
            topic=props.get("topic", ""),
            country=props.get("country", ""),
            active=props.get("active", True),
            last_modified=props.get("last_modified", ""),
            score=obj.metadata.score if hasattr(obj.metadata, "score") and obj.metadata.score else 0.0,
        )

    def close(self):
        """Close Weaviate client connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate client closed")


# Backwards compatibility alias
HybridRetriever = EnhancedHybridRetriever
