"""Enhanced Hybrid Retriever with adaptive behavior and fused ranking.

Features:
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
from langfuse import observe
import json
import os
from datetime import datetime
from dataclasses import asdict
import uuid

from .models import PolicyChunk, QueryType, RetrievalResult
from .config import RetrievalConfig
from .tools import serve_chunks

logger = logging.getLogger(__name__)


class EnhancedHybridRetriever:
    """Production-grade hybrid retriever with adaptive behavior.

    Features:
    - Query classification for optimal BM25/vector balance
    - CrossEncoder reranking with calibrated confidence scores
    - Optional Reciprocal Rank Fusion (RRF) for stable ranking
    - Batch fetching of context windows
    - Confidence-based filtering
    """

    def __init__(self, config: RetrievalConfig = None):
        """Initialize the EnhancedHybridRetriever.

        Args:
            config: Retrieval configuration object
        """
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

        # Ensure debug directory exists
        if self.config.debug_to_file:
            os.makedirs(self.config.retrieval_output_dir, exist_ok=True)
            logger.info(f"Debug logging enabled. Output dir: {self.config.retrieval_output_dir}")

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
        """Classify query type for adaptive alpha selection.

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
        """Get optimal alpha based on query type.

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

    def _dump_debug_json(
        self, request_id: str, step: str, query: str, chunks: List[PolicyChunk], metadata: Dict[str, Any] = None
    ):
        """Dump retrieval state to JSON file for debugging."""
        if not self.config.debug_to_file:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{request_id}_{step}.json"
            path = os.path.join(self.config.retrieval_output_dir, filename)

            # Convert chunks to dicts
            chunks_data = [asdict(c) for c in chunks]

            data = {
                "request_id": request_id,
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "metadata": metadata or {},
                "chunk_count": len(chunks),
                "chunks": chunks_data,
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to dump debug JSON: {e}")

    @observe(as_type="generation")
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        override_alpha: Optional[float] = None,
    ) -> List[PolicyChunk]:
        """Retrieve relevant policy chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"country": "CH"})
            override_alpha: Override adaptive alpha selection

        Returns:
            List of PolicyChunk objects with confidence scores
        """
        top_k = top_k or self.config.default_top_k
        request_id = str(uuid.uuid4())[:8]

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
            logger.warning("No results found via initial search")
            return []

        logger.info(f"Retrieved {len(chunks)} candidates via hybrid search")

        # 1. Populate initial metadata immediately (query_type, alpha)
        # This fixes missing data in "1_retrieval" debug step
        for chunk in chunks:
            chunk.query_type = query_type.value
            # We don't have per-chunk alpha if using RRF, but if hybrid, we do.
            # If RRF, alpha is not used in the same way, but we can leave null or set global.
            if not self.config.enable_rrf:
                chunk.alpha_used = alpha

            # Initialize confidence/ranking fields to pending/unknown instead of null if preferred,
            # but cleaning up the nulls in JSON was the user request.
            # We'll leave them as is (None) but now we know why they are None at this stage.

        # 2. Serve chunks (Text) immediately to provide context even for candidates
        # This addresses "monitoring with more data" request.
        if self.config.enable_context_window:
            serve_chunks(self.client, chunks)

        self._dump_debug_json(
            request_id,
            "1_retrieval",
            query,
            chunks,
            {
                "method": "rrf" if self.config.enable_rrf else "hybrid",
                "alpha": alpha,
                "candidate_size": candidate_size,
            },
        )

        # Rerank if enabled
        if self.config.enable_reranking and self.reranker:
            chunks = self._rerank_chunks(query, chunks, top_k)
            logger.info(f"After reranking (CrossEncoder) (top_k={top_k}): {len(chunks)} chunks kept")
        else:
            chunks = chunks[:top_k]
            logger.info(f"No reranking (top_k={top_k}): {len(chunks)} chunks kept")

        self._dump_debug_json(
            request_id,
            "2_reranking",
            query,
            chunks,
            {"reranking_enabled": bool(self.config.enable_reranking), "top_k": top_k},
        )

        # Assign confidence levels
        self._assign_confidence_levels(chunks)

        # Log confidence distribution
        conf_counts = {"high": 0, "medium": 0, "low": 0}
        for c in chunks:
            conf_counts[c.confidence_level] = conf_counts.get(c.confidence_level, 0) + 1
        logger.info(f"Confidence levels: {conf_counts}")

        # Filter low confidence (but keep at least one result)
        confident_chunks = [c for c in chunks if c.confidence_level != "low"]

        if not confident_chunks and chunks:
            logger.info("All chunks low confidence - keeping best one as 'low_but_best'")
            chunks[0].confidence_level = "low_but_best"
            return [chunks[0]]

        logger.info(
            f"Final output: {len(confident_chunks)} chunks (filtered {len(chunks) - len(confident_chunks)} low confidence)"
        )

        self._dump_debug_json(
            request_id,
            "3_final_output",
            query,
            confident_chunks,
            {"filtered_count": len(chunks) - len(confident_chunks)},
        )
        return confident_chunks

    @observe(as_type="generation")
    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Retrieve with full metadata for observability.

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
            serve_chunks(self.client, chunks)

        # Metadata already assigned above in retrieve? No, retrieve_with_metadata calls _retrieve_hybrid directly.
        # We need to assign it here too.
        for chunk in chunks:
            chunk.query_type = query_type.value
            chunk.alpha_used = alpha

        return RetrievalResult(
            chunks=chunks,
            query=query,
            query_type=query_type,
            alpha_used=alpha,
            total_candidates=total_candidates,
            filters_applied=filters,
        )

    def _retrieve_hybrid(self, query: str, limit: int, filters: Dict[str, Any], alpha: float) -> List[PolicyChunk]:
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

    def _retrieve_with_rrf(self, query: str, limit: int, filters: Dict[str, Any]) -> List[PolicyChunk]:
        """Reciprocal Rank Fusion: run BM25 and vector separately, then fuse.

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

    @observe(as_type="generation")
    def _rerank_chunks(self, query: str, chunks: List[PolicyChunk], top_k: int) -> List[PolicyChunk]:
        """Rerank using CrossEncoder with calibrated scores.

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

        # Log top 3 scores for debugging
        top_scores = [f"{c.score:.4f}" for c in chunks[:3]]
        logger.info(f"Top rerank scores: {top_scores}")

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
            section_path=props.get("section_path", []) or [props.get("section_path_str", "")],
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
