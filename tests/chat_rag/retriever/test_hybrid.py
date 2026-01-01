"""
Tests for EnhancedHybridRetriever.
"""

import pytest
import numpy as np
import sys
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from pathlib import Path
import tempfile
import json

# Mock weaviate before imports
sys.modules["weaviate"] = MagicMock()
sys.modules["weaviate.classes"] = MagicMock()
sys.modules["weaviate.classes.query"] = MagicMock()
sys.modules["weaviate.classes.config"] = MagicMock()

from src.chat_rag.retriever.hybrid import EnhancedHybridRetriever
from src.chat_rag.retriever.models import PolicyChunk, QueryType, RetrievalResult
from src.chat_rag.retriever.config import RetrievalConfig


class TestQueryClassification:
    """Tests for query classification logic."""

    @pytest.fixture
    def retriever(self):
        """Create retriever with mocked dependencies."""
        with patch("src.chat_rag.retriever.hybrid.weaviate") as mock_weaviate, \
             patch("src.chat_rag.retriever.hybrid.OpenAIEmbedding") as mock_embed:

            mock_client = Mock()
            mock_client.is_connected.return_value = True
            mock_weaviate.connect_to_local.return_value = mock_client

            config = RetrievalConfig(weaviate_url="http://localhost:8080")
            retriever = EnhancedHybridRetriever(config=config)
            yield retriever
            retriever.close()

    def test_classify_factual_queries(self, retriever):
        """Test classification of factual queries."""
        factual_queries = [
            "How many vacation days do I get?",
            "What is the deadline for submitting expenses?",
            "What is the maximum amount for meals?",
            "When can I take parental leave?",
            "How much does the company contribute?",
            "What is the rate for overtime?",
        ]

        for query in factual_queries:
            query_type = retriever.classify_query(query)
            assert query_type == QueryType.FACTUAL, f"Query '{query}' should be FACTUAL, got {query_type}"

    def test_classify_procedural_queries(self, retriever):
        """Test classification of procedural queries."""
        procedural_queries = [
            "How do I request time off?",
            "How to submit an expense claim?",
            "What are the steps to enroll in benefits?",
            "How do I apply for remote work?",
        ]

        for query in procedural_queries:
            query_type = retriever.classify_query(query)
            assert query_type == QueryType.PROCEDURAL, f"Query '{query}' should be PROCEDURAL, got {query_type}"

    def test_classify_conceptual_queries(self, retriever):
        """Test classification of conceptual queries."""
        conceptual_queries = [
            "Tell me about the company benefits",
            "Explain the vacation policy",
            "What benefits does the company offer?",
            "Can you describe our health insurance?",
        ]

        for query in conceptual_queries:
            query_type = retriever.classify_query(query)
            assert query_type == QueryType.CONCEPTUAL, f"Query '{query}' should be CONCEPTUAL, got {query_type}"

    def test_classify_unknown_queries(self, retriever):
        """Test classification of ambiguous queries."""
        # Queries without clear indicators should be CONCEPTUAL (default)
        query = "vacation"
        query_type = retriever.classify_query(query)
        assert query_type == QueryType.CONCEPTUAL


class TestAdaptiveAlpha:
    """Tests for adaptive alpha selection."""

    @pytest.fixture
    def retriever(self):
        """Create retriever with mocked dependencies."""
        with patch("src.chat_rag.retriever.hybrid.weaviate") as mock_weaviate, \
             patch("src.chat_rag.retriever.hybrid.OpenAIEmbedding") as mock_embed:

            mock_client = Mock()
            mock_client.is_connected.return_value = True
            mock_weaviate.connect_to_local.return_value = mock_client

            config = RetrievalConfig(
                weaviate_url="http://localhost:8080",
                alpha_factual=0.3,
                alpha_conceptual=0.8,
                alpha_procedural=0.6,
                alpha_default=0.5,
            )
            retriever = EnhancedHybridRetriever(config=config)
            yield retriever
            retriever.close()

    def test_get_alpha_for_factual(self, retriever):
        """Test alpha selection for factual queries."""
        alpha = retriever.get_adaptive_alpha(QueryType.FACTUAL)
        assert alpha == 0.3

    def test_get_alpha_for_conceptual(self, retriever):
        """Test alpha selection for conceptual queries."""
        alpha = retriever.get_adaptive_alpha(QueryType.CONCEPTUAL)
        assert alpha == 0.8

    def test_get_alpha_for_procedural(self, retriever):
        """Test alpha selection for procedural queries."""
        alpha = retriever.get_adaptive_alpha(QueryType.PROCEDURAL)
        assert alpha == 0.6

    def test_get_alpha_for_unknown(self, retriever):
        """Test alpha selection for unknown queries."""
        alpha = retriever.get_adaptive_alpha(QueryType.UNKNOWN)
        assert alpha == 0.5


class TestFilterBuilding:
    """Tests for Weaviate filter construction."""

    @pytest.fixture
    def retriever(self):
        """Create retriever with mocked dependencies."""
        with patch("src.chat_rag.retriever.hybrid.weaviate") as mock_weaviate, \
             patch("src.chat_rag.retriever.hybrid.OpenAIEmbedding") as mock_embed:

            mock_client = Mock()
            mock_client.is_connected.return_value = True
            mock_weaviate.connect_to_local.return_value = mock_client

            config = RetrievalConfig(weaviate_url="http://localhost:8080")
            retriever = EnhancedHybridRetriever(config=config)
            yield retriever
            retriever.close()

    def test_build_empty_filters(self, retriever):
        """Test building filters from empty dict."""
        weaviate_filter = retriever._build_filters({})
        assert weaviate_filter is None

    def test_build_single_filter(self, retriever):
        """Test building single filter."""
        filters = {"active": True}
        weaviate_filter = retriever._build_filters(filters)
        assert weaviate_filter is not None

    def test_build_multiple_filters(self, retriever):
        """Test building multiple filters."""
        filters = {"active": True, "country": "CH"}
        weaviate_filter = retriever._build_filters(filters)
        assert weaviate_filter is not None

    def test_build_filters_with_various_types(self, retriever):
        """Test building filters with different value types."""
        filters = {
            "active": True,
            "country": "CH",
            "topic": "HR",
        }
        weaviate_filter = retriever._build_filters(filters)
        assert weaviate_filter is not None


class TestConfidenceAssignment:
    """Tests for confidence level assignment."""

    @pytest.fixture
    def retriever(self):
        """Create retriever with mocked dependencies."""
        with patch("src.chat_rag.retriever.hybrid.weaviate") as mock_weaviate, \
             patch("src.chat_rag.retriever.hybrid.OpenAIEmbedding") as mock_embed:

            mock_client = Mock()
            mock_client.is_connected.return_value = True
            mock_weaviate.connect_to_local.return_value = mock_client

            config = RetrievalConfig(
                weaviate_url="http://localhost:8080",
                high_confidence_threshold=0.7,
                medium_confidence_threshold=0.5,
            )
            retriever = EnhancedHybridRetriever(config=config)
            yield retriever
            retriever.close()

    def test_assign_high_confidence(self, retriever):
        """Test assigning high confidence level."""
        chunk = PolicyChunk(
            text="test",
            document_id="doc1",
            document_name="Doc",
            section_path_str="Section",
            section_path=["Section"],
            chunk_id="chunk1",
            chunk_index=0,
            topic="Test",
            country="CH",
            active=True,
            last_modified="2024-01-01",
            score=0.85,
        )

        retriever._assign_confidence_levels([chunk])
        assert chunk.confidence_level == "high"

    def test_assign_medium_confidence(self, retriever):
        """Test assigning medium confidence level."""
        chunk = PolicyChunk(
            text="test",
            document_id="doc1",
            document_name="Doc",
            section_path_str="Section",
            section_path=["Section"],
            chunk_id="chunk1",
            chunk_index=0,
            topic="Test",
            country="CH",
            active=True,
            last_modified="2024-01-01",
            score=0.6,
        )

        retriever._assign_confidence_levels([chunk])
        assert chunk.confidence_level == "medium"

    def test_assign_low_confidence(self, retriever):
        """Test assigning low confidence level."""
        chunk = PolicyChunk(
            text="test",
            document_id="doc1",
            document_name="Doc",
            section_path_str="Section",
            section_path=["Section"],
            chunk_id="chunk1",
            chunk_index=0,
            topic="Test",
            country="CH",
            active=True,
            last_modified="2024-01-01",
            score=0.3,
        )

        retriever._assign_confidence_levels([chunk])
        assert chunk.confidence_level == "low"

    def test_assign_confidence_at_threshold_boundary(self, retriever):
        """Test confidence assignment at exact threshold values."""
        # Exactly at high threshold
        chunk_high = PolicyChunk(
            text="test",
            document_id="doc1",
            document_name="Doc",
            section_path_str="Section",
            section_path=["Section"],
            chunk_id="chunk1",
            chunk_index=0,
            topic="Test",
            country="CH",
            active=True,
            last_modified="2024-01-01",
            score=0.7,
        )
        retriever._assign_confidence_levels([chunk_high])
        assert chunk_high.confidence_level == "high"

        # Exactly at medium threshold
        chunk_medium = PolicyChunk(
            text="test",
            document_id="doc2",
            document_name="Doc",
            section_path_str="Section",
            section_path=["Section"],
            chunk_id="chunk2",
            chunk_index=0,
            topic="Test",
            country="CH",
            active=True,
            last_modified="2024-01-01",
            score=0.5,
        )
        retriever._assign_confidence_levels([chunk_medium])
        assert chunk_medium.confidence_level == "medium"

    def test_assign_confidence_to_multiple_chunks(self, retriever):
        """Test assigning confidence to multiple chunks."""
        chunks = [
            PolicyChunk(
                text=f"test{i}",
                document_id=f"doc{i}",
                document_name="Doc",
                section_path_str="Section",
                section_path=["Section"],
                chunk_id=f"chunk{i}",
                chunk_index=i,
                topic="Test",
                country="CH",
                active=True,
                last_modified="2024-01-01",
                score=score,
            )
            for i, score in enumerate([0.9, 0.6, 0.3])
        ]

        retriever._assign_confidence_levels(chunks)
        assert chunks[0].confidence_level == "high"
        assert chunks[1].confidence_level == "medium"
        assert chunks[2].confidence_level == "low"


class TestRerankingWithCrossEncoder:
    """Tests for reranking with CrossEncoder."""

    @pytest.fixture
    def retriever(self):
        """Create retriever with mocked CrossEncoder."""
        with patch("src.chat_rag.retriever.hybrid.weaviate") as mock_weaviate, \
             patch("src.chat_rag.retriever.hybrid.OpenAIEmbedding") as mock_embed, \
             patch("src.chat_rag.retriever.hybrid.CrossEncoder") as mock_cross_encoder:

            mock_client = Mock()
            mock_client.is_connected.return_value = True
            mock_weaviate.connect_to_local.return_value = mock_client

            config = RetrievalConfig(
                weaviate_url="http://localhost:8080",
                enable_reranking=True,
            )
            retriever = EnhancedHybridRetriever(config=config)

            # Mock CrossEncoder predict method
            mock_cross_encoder_instance = mock_cross_encoder.return_value
            retriever.reranker = mock_cross_encoder_instance

            yield retriever, mock_cross_encoder_instance
            retriever.close()

    def test_rerank_chunks_basic(self, retriever):
        """Test basic reranking functionality."""
        retriever_obj, mock_reranker = retriever

        # Mock predict to return scores
        mock_reranker.predict.return_value = np.array([0.2, 0.8, 0.5])

        chunks = [
            PolicyChunk(
                text=f"chunk{i}",
                document_id=f"doc{i}",
                document_name="Doc",
                section_path_str="Section",
                section_path=["Section"],
                chunk_id=f"chunk{i}",
                chunk_index=i,
                topic="Test",
                country="CH",
                active=True,
                last_modified="2024-01-01",
                score=0.5,
            )
            for i in range(3)
        ]

        reranked = retriever_obj._rerank_chunks("test query", chunks, top_k=2)

        # Should be sorted by rerank score (descending)
        assert len(reranked) == 2
        assert reranked[0].text == "chunk1"  # Highest score (0.8)
        assert reranked[1].text == "chunk2"  # Second highest (0.5)

        # Scores should be calibrated (sigmoid applied)
        assert reranked[0].score > 0.5  # Calibrated score
        assert reranked[0].rerank_score is not None

    def test_rerank_empty_chunks(self, retriever):
        """Test reranking with empty chunk list."""
        retriever_obj, _ = retriever
        reranked = retriever_obj._rerank_chunks("test query", [], top_k=5)
        assert reranked == []

    def test_rerank_sigmoid_calibration(self, retriever):
        """Test that sigmoid calibration is applied to raw scores."""
        retriever_obj, mock_reranker = retriever

        # Raw scores that will be calibrated
        raw_scores = np.array([2.0, -2.0, 0.0])
        mock_reranker.predict.return_value = raw_scores

        chunks = [
            PolicyChunk(
                text=f"chunk{i}",
                document_id=f"doc{i}",
                document_name="Doc",
                section_path_str="Section",
                section_path=["Section"],
                chunk_id=f"chunk{i}",
                chunk_index=i,
                topic="Test",
                country="CH",
                active=True,
                last_modified="2024-01-01",
                score=0.5,
            )
            for i in range(3)
        ]

        reranked = retriever_obj._rerank_chunks("test query", chunks, top_k=3)

        # Check sigmoid calibration was applied
        # sigmoid(2.0) ≈ 0.88, sigmoid(-2.0) ≈ 0.12, sigmoid(0.0) = 0.5
        assert 0.85 < reranked[0].score < 0.90  # chunk0 had raw score 2.0
        assert 0.45 < reranked[1].score < 0.55  # chunk2 had raw score 0.0
        assert 0.10 < reranked[2].score < 0.15  # chunk1 had raw score -2.0


class TestRRFRetrieval:
    """Tests for Reciprocal Rank Fusion."""

    @pytest.fixture
    def retriever(self):
        """Create retriever with RRF enabled."""
        with patch("src.chat_rag.retriever.hybrid.weaviate") as mock_weaviate, \
             patch("src.chat_rag.retriever.hybrid.OpenAIEmbedding") as mock_embed:

            mock_client = Mock()
            mock_client.is_connected.return_value = True

            # Mock collection
            mock_collection = Mock()
            mock_client.collections.get.return_value = mock_collection

            mock_weaviate.connect_to_local.return_value = mock_client

            # Mock embed model
            mock_embed_instance = Mock()
            mock_embed_instance.get_query_embedding.return_value = [0.1] * 1536
            mock_embed.return_value = mock_embed_instance

            config = RetrievalConfig(
                weaviate_url="http://localhost:8080",
                enable_rrf=True,
                rrf_k=60,
            )
            retriever = EnhancedHybridRetriever(config=config)

            yield retriever, mock_collection
            retriever.close()

    def test_rrf_fusion_basic(self, retriever):
        """Test basic RRF fusion of BM25 and vector results."""
        retriever_obj, mock_collection = retriever

        # Create mock objects for BM25 results
        bm25_obj1 = Mock()
        bm25_obj1.properties = {
            "chunk_id": "chunk1",
            "text": "text1",
            "document_id": "doc1",
            "document_name": "Doc1",
            "section_path_str": "Section",
            "chunk_index": 0,
            "topic": "Test",
            "country": "CH",
            "active": True,
            "last_modified": "2024-01-01",
        }
        bm25_obj1.metadata = Mock(score=None)

        bm25_obj2 = Mock()
        bm25_obj2.properties = {
            "chunk_id": "chunk2",
            "text": "text2",
            "document_id": "doc2",
            "document_name": "Doc2",
            "section_path_str": "Section",
            "chunk_index": 0,
            "topic": "Test",
            "country": "CH",
            "active": True,
            "last_modified": "2024-01-01",
        }
        bm25_obj2.metadata = Mock(score=None)

        # Create mock objects for vector results
        vector_obj1 = Mock()
        vector_obj1.properties = {
            "chunk_id": "chunk2",  # Same as bm25_obj2 (should get boosted)
            "text": "text2",
            "document_id": "doc2",
            "document_name": "Doc2",
            "section_path_str": "Section",
            "chunk_index": 0,
            "topic": "Test",
            "country": "CH",
            "active": True,
            "last_modified": "2024-01-01",
        }
        vector_obj1.metadata = Mock(score=None)

        vector_obj3 = Mock()
        vector_obj3.properties = {
            "chunk_id": "chunk3",
            "text": "text3",
            "document_id": "doc3",
            "document_name": "Doc3",
            "section_path_str": "Section",
            "chunk_index": 0,
            "topic": "Test",
            "country": "CH",
            "active": True,
            "last_modified": "2024-01-01",
        }
        vector_obj3.metadata = Mock(score=None)

        # Mock BM25 and vector search responses
        bm25_response = Mock()
        bm25_response.objects = [bm25_obj1, bm25_obj2]

        vector_response = Mock()
        vector_response.objects = [vector_obj1, vector_obj3]

        mock_collection.query.bm25.return_value = bm25_response
        mock_collection.query.near_vector.return_value = vector_response

        # Execute RRF retrieval
        chunks = retriever_obj._retrieve_with_rrf(
            query="test query",
            limit=10,
            filters={"active": True}
        )

        # chunk2 appears in both results (rank 1 in BM25, rank 0 in vector)
        # So it should have highest RRF score
        assert len(chunks) > 0
        assert chunks[0].chunk_id == "chunk2"

        # Check that ranks are recorded
        chunk2 = chunks[0]
        assert chunk2.bm25_rank == 1
        assert chunk2.vector_rank == 0

    def test_rrf_k_parameter(self, retriever):
        """Test that RRF k parameter is used correctly."""
        retriever_obj, mock_collection = retriever

        # Single chunk appearing at rank 0 in BM25
        bm25_obj = Mock()
        bm25_obj.properties = {
            "chunk_id": "chunk1",
            "text": "text1",
            "document_id": "doc1",
            "document_name": "Doc1",
            "section_path_str": "Section",
            "chunk_index": 0,
            "topic": "Test",
            "country": "CH",
            "active": True,
            "last_modified": "2024-01-01",
        }
        bm25_obj.metadata = Mock(score=None)

        bm25_response = Mock()
        bm25_response.objects = [bm25_obj]

        vector_response = Mock()
        vector_response.objects = []

        mock_collection.query.bm25.return_value = bm25_response
        mock_collection.query.near_vector.return_value = vector_response

        chunks = retriever_obj._retrieve_with_rrf(
            query="test",
            limit=10,
            filters={}
        )

        # RRF score = 1/(k + rank + 1) = 1/(60 + 0 + 1) ≈ 0.0164
        assert len(chunks) == 1
        assert abs(chunks[0].score - 1/61) < 0.001


class TestDebugFileWriting:
    """Tests for debug file writing functionality."""

    @pytest.fixture
    def retriever_with_debug(self):
        """Create retriever with debug file writing enabled."""
        with patch("src.chat_rag.retriever.hybrid.weaviate") as mock_weaviate, \
             patch("src.chat_rag.retriever.hybrid.OpenAIEmbedding") as mock_embed, \
             tempfile.TemporaryDirectory() as tmpdir:

            mock_client = Mock()
            mock_client.is_connected.return_value = True
            mock_weaviate.connect_to_local.return_value = mock_client

            config = RetrievalConfig(
                weaviate_url="http://localhost:8080",
                debug_to_file=True,
                retrieval_output_dir=tmpdir,
            )
            retriever = EnhancedHybridRetriever(config=config)
            yield retriever, Path(tmpdir)
            retriever.close()

    def test_debug_json_created(self, retriever_with_debug):
        """Test that debug JSON files are created."""
        retriever, tmpdir = retriever_with_debug

        chunk = PolicyChunk(
            text="test",
            document_id="doc1",
            document_name="Doc",
            section_path_str="Section",
            section_path=["Section"],
            chunk_id="chunk1",
            chunk_index=0,
            topic="Test",
            country="CH",
            active=True,
            last_modified="2024-01-01",
            score=0.8,
        )

        retriever._dump_debug_json(
            request_id="test123",
            step="test_step",
            query="test query",
            chunks=[chunk],
            metadata={"test_key": "test_value"}
        )

        # Check that a JSON file was created
        json_files = list(Path(tmpdir).glob("*.json"))
        assert len(json_files) == 1

        # Verify JSON content
        with open(json_files[0]) as f:
            data = json.load(f)
            assert data["request_id"] == "test123"
            assert data["step"] == "test_step"
            assert data["query"] == "test query"
            assert data["chunk_count"] == 1
            assert len(data["chunks"]) == 1
            assert data["metadata"]["test_key"] == "test_value"

    def test_debug_disabled(self):
        """Test that debug files are not created when disabled."""
        with patch("src.chat_rag.retriever.hybrid.weaviate") as mock_weaviate, \
             patch("src.chat_rag.retriever.hybrid.OpenAIEmbedding") as mock_embed, \
             tempfile.TemporaryDirectory() as tmpdir:

            mock_client = Mock()
            mock_client.is_connected.return_value = True
            mock_weaviate.connect_to_local.return_value = mock_client

            config = RetrievalConfig(
                weaviate_url="http://localhost:8080",
                debug_to_file=False,
                retrieval_output_dir=tmpdir,
            )
            retriever = EnhancedHybridRetriever(config=config)

            chunk = PolicyChunk(
                text="test",
                document_id="doc1",
                document_name="Doc",
                section_path_str="Section",
                section_path=["Section"],
                chunk_id="chunk1",
                chunk_index=0,
                topic="Test",
                country="CH",
                active=True,
                last_modified="2024-01-01",
                score=0.8,
            )

            retriever._dump_debug_json(
                request_id="test123",
                step="test_step",
                query="test query",
                chunks=[chunk],
            )

            # No JSON files should be created
            json_files = list(Path(tmpdir).glob("*.json"))
            assert len(json_files) == 0

            retriever.close()


class TestPolicyChunkConversion:
    """Tests for converting Weaviate objects to PolicyChunk."""

    @pytest.fixture
    def retriever(self):
        """Create retriever instance."""
        with patch("src.chat_rag.retriever.hybrid.weaviate") as mock_weaviate, \
             patch("src.chat_rag.retriever.hybrid.OpenAIEmbedding") as mock_embed:

            mock_client = Mock()
            mock_client.is_connected.return_value = True
            mock_weaviate.connect_to_local.return_value = mock_client

            config = RetrievalConfig(weaviate_url="http://localhost:8080")
            retriever = EnhancedHybridRetriever(config=config)
            yield retriever
            retriever.close()

    def test_to_policy_chunk_basic(self, retriever):
        """Test basic conversion from Weaviate object to PolicyChunk."""
        mock_obj = Mock()
        mock_obj.properties = {
            "text": "test text",
            "document_id": "doc1",
            "document_name": "Test Doc",
            "section_path_str": "Section > Subsection",
            "section_path": ["Section", "Subsection"],
            "chunk_id": "chunk1",
            "chunk_index": 5,
            "topic": "HR",
            "country": "CH",
            "active": True,
            "last_modified": "2024-01-01",
        }
        mock_obj.metadata = Mock(score=0.85)

        chunk = retriever._to_policy_chunk(mock_obj)

        assert isinstance(chunk, PolicyChunk)
        assert chunk.text == "test text"
        assert chunk.document_id == "doc1"
        assert chunk.document_name == "Test Doc"
        assert chunk.section_path_str == "Section > Subsection"
        assert chunk.section_path == ["Section", "Subsection"]
        assert chunk.chunk_id == "chunk1"
        assert chunk.chunk_index == 5
        assert chunk.topic == "HR"
        assert chunk.country == "CH"
        assert chunk.active is True
        assert chunk.last_modified == "2024-01-01"
        assert chunk.score == 0.85

    def test_to_policy_chunk_missing_section_path(self, retriever):
        """Test conversion when section_path is missing."""
        mock_obj = Mock()
        mock_obj.properties = {
            "text": "test",
            "document_id": "doc1",
            "document_name": "Doc",
            "section_path_str": "Section",
            "chunk_id": "chunk1",
            "chunk_index": 0,
            "topic": "Test",
            "country": "CH",
            "active": True,
            "last_modified": "2024-01-01",
        }
        mock_obj.metadata = Mock(score=0.5)

        chunk = retriever._to_policy_chunk(mock_obj)

        # Should fallback to using section_path_str as a single-element list
        assert chunk.section_path == ["Section"]

    def test_to_policy_chunk_missing_score(self, retriever):
        """Test conversion when score is missing."""
        mock_obj = Mock()
        mock_obj.properties = {
            "text": "test",
            "document_id": "doc1",
            "document_name": "Doc",
            "section_path_str": "Section",
            "chunk_id": "chunk1",
            "chunk_index": 0,
            "topic": "Test",
            "country": "CH",
            "active": True,
            "last_modified": "2024-01-01",
        }
        mock_obj.metadata = Mock(spec=[])  # No score attribute

        chunk = retriever._to_policy_chunk(mock_obj)

        # Should default to 0.0
        assert chunk.score == 0.0
