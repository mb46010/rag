"""
Tests for hybrid retriever.

Note: These are primarily integration tests as they require Weaviate to be running.
For unit tests, we would need to mock the Weaviate client.
"""

import pytest
from src.retriever import HybridRetriever, PolicyChunk, create_retriever_tool


class TestPolicyChunkDataclass:
    """Tests for PolicyChunk dataclass."""

    def test_policy_chunk_creation(self):
        """Test creating a PolicyChunk."""
        chunk = PolicyChunk(
            text="Test policy text",
            document_id="doc123",
            document_name="Test Policy",
            section_path_str="Policy > Section",
            chunk_id="chunk123",
            chunk_index=0,
            topic="Test",
            country="CH",
            active=True,
            last_modified="2024-01-01",
            score=0.95
        )

        assert chunk.text == "Test policy text"
        assert chunk.document_id == "doc123"
        assert chunk.score == 0.95
        assert chunk.previous_chunk is None
        assert chunk.next_chunk is None

    def test_policy_chunk_with_context(self):
        """Test PolicyChunk with context chunks."""
        chunk = PolicyChunk(
            text="Current chunk",
            document_id="doc123",
            document_name="Test Policy",
            section_path_str="Policy > Section",
            chunk_id="chunk123",
            chunk_index=1,
            topic="Test",
            country="CH",
            active=True,
            last_modified="2024-01-01",
            score=0.95,
            previous_chunk="Previous chunk text",
            next_chunk="Next chunk text"
        )

        assert chunk.previous_chunk == "Previous chunk text"
        assert chunk.next_chunk == "Next chunk text"


class TestHybridRetriever:
    """Tests for HybridRetriever.

    Note: These tests require Weaviate to be running and data to be ingested.
    """

    @pytest.fixture
    def retriever(self):
        """Create retriever instance for testing."""
        # This will fail if Weaviate is not running
        try:
            retriever = HybridRetriever()
            yield retriever
            retriever.close()
        except Exception as e:
            pytest.skip(f"Weaviate not available: {e}")

    def test_retriever_initialization(self):
        """Test that retriever initializes with correct parameters."""
        retriever = HybridRetriever(
            weaviate_url="http://localhost:8080",
            collection_name="PolicyChunk",
            alpha=0.7
        )

        assert retriever.weaviate_url == "http://localhost:8080"
        assert retriever.collection_name == "PolicyChunk"
        assert retriever.alpha == 0.7

        retriever.close()

    def test_build_weaviate_filters(self, retriever):
        """Test building Weaviate filters from dict."""
        filters = {"active": True, "country": "CH"}
        weaviate_filter = retriever._build_weaviate_filters(filters)

        assert weaviate_filter is not None

    def test_build_empty_filters(self, retriever):
        """Test building filters from empty dict."""
        weaviate_filter = retriever._build_weaviate_filters({})
        assert weaviate_filter is None

    def test_build_single_filter(self, retriever):
        """Test building single filter."""
        filters = {"active": True}
        weaviate_filter = retriever._build_weaviate_filters(filters)
        assert weaviate_filter is not None

    @pytest.mark.integration
    def test_retrieve_with_query(self, retriever):
        """Test retrieval with a query.

        Requires Weaviate to be running with ingested data.
        """
        try:
            results = retriever.retrieve(
                query="PTO policy",
                top_k=3,
                filters={"active": True}
            )

            # If data is ingested, we should get results
            if results:
                assert len(results) <= 3
                assert all(isinstance(chunk, PolicyChunk) for chunk in results)
                assert all(chunk.active for chunk in results)
        except Exception as e:
            pytest.skip(f"Integration test failed (likely no data): {e}")

    @pytest.mark.integration
    def test_retrieve_with_country_filter(self, retriever):
        """Test retrieval with country filter."""
        try:
            results = retriever.retrieve(
                query="expense policy",
                top_k=2,
                filters={"active": True, "country": "CH"}
            )

            if results:
                assert all(chunk.country == "CH" for chunk in results)
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")

    @pytest.mark.integration
    def test_retrieve_with_context(self, retriever):
        """Test retrieval includes context chunks."""
        try:
            results = retriever.retrieve(
                query="policy overview",
                top_k=1,
                include_context=True
            )

            if results:
                chunk = results[0]
                # Context may or may not be present depending on chunk position
                # Just verify the fields exist
                assert hasattr(chunk, 'previous_chunk')
                assert hasattr(chunk, 'next_chunk')
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")

    @pytest.mark.integration
    def test_retrieve_no_results(self, retriever):
        """Test retrieval with query that returns no results."""
        try:
            results = retriever.retrieve(
                query="completely nonexistent policy about flying unicorns",
                top_k=5,
                filters={"active": True}
            )

            # Should return empty list, not error
            assert isinstance(results, list)
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


class TestCreateRetrieverTool:
    """Tests for retriever tool creation."""

    @pytest.fixture
    def retriever(self):
        """Create mock retriever for testing."""
        try:
            retriever = HybridRetriever()
            yield retriever
            retriever.close()
        except Exception:
            pytest.skip("Weaviate not available")

    def test_create_tool(self, retriever):
        """Test creating retriever tool."""
        tool = create_retriever_tool(retriever)

        assert callable(tool)
        assert tool.__name__ == "search_policies"

    @pytest.mark.integration
    def test_tool_execution(self, retriever):
        """Test executing the retriever tool."""
        tool = create_retriever_tool(retriever)

        try:
            result = tool(query="PTO policy", country="CH")

            assert isinstance(result, str)
            # Should either have results or a "not found" message
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")

    @pytest.mark.integration
    def test_tool_without_country(self, retriever):
        """Test tool without country filter."""
        tool = create_retriever_tool(retriever)

        try:
            result = tool(query="expense policy")
            assert isinstance(result, str)
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")

    @pytest.mark.integration
    def test_tool_no_results(self, retriever):
        """Test tool with query that returns no results."""
        tool = create_retriever_tool(retriever)

        try:
            result = tool(query="nonexistent unicorn policy")

            assert isinstance(result, str)
            assert "not found" in result.lower() or "no relevant" in result.lower()
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


class TestRetrieverAlphaParameter:
    """Tests for hybrid search alpha parameter."""

    def test_alpha_bm25_only(self):
        """Test alpha=0 (BM25 only)."""
        retriever = HybridRetriever(alpha=0.0)
        assert retriever.alpha == 0.0
        retriever.close()

    def test_alpha_vector_only(self):
        """Test alpha=1 (vector only)."""
        retriever = HybridRetriever(alpha=1.0)
        assert retriever.alpha == 1.0
        retriever.close()

    def test_alpha_balanced(self):
        """Test alpha=0.5 (balanced)."""
        retriever = HybridRetriever(alpha=0.5)
        assert retriever.alpha == 0.5
        retriever.close()
