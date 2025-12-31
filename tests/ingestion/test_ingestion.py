"""
Tests for ingestion pipeline.
"""

import pytest
import json
from pathlib import Path
from src.ingestion import IngestionPipeline, PolicyChunk
from src.ingestion.text_processing import (
    split_by_headers,
    generate_document_id,
    generate_chunk_id,
    count_tokens,
    create_policy_chunk,
)


class TestIngestionPipeline:
    """Tests for IngestionPipeline."""

    @pytest.fixture
    def sample_document(self):
        """Sample policy document for testing."""
        return {
            "metadata": {
                "document_id": "pol_test",
                "document_name": "Test Policy",
                "topic": "Testing",
                "country": "CH",
                "active": True,
                "last_modified": "2024-01-01",
            },
            "content": """# Test Policy

## 1. Overview
This is a test policy for unit testing.

## 2. Details
### 2.1 Subsection
This is a subsection with more details.

### 2.2 Another Subsection
This is another subsection.

## 3. Conclusion
This is the conclusion.""",
        }

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create a test pipeline instance with mocked storage."""
        from unittest.mock import MagicMock, patch

        with patch("src.ingestion.pipeline.WeaviateStorage") as MockStorage:
            # Configure mock
            mock_storage_instance = MagicMock()
            MockStorage.return_value = mock_storage_instance

            pipeline = IngestionPipeline(docs_dir=str(tmp_path))
            return pipeline

    def test_split_by_headers(self, sample_document):
        """Test splitting content by markdown headers."""
        content = sample_document["content"]
        sections = split_by_headers(content)

        assert len(sections) > 0

        # Check that sections have paths
        for section_path, text in sections:
            assert isinstance(section_path, list)
            assert len(section_path) > 0
            assert isinstance(text, str)
            assert len(text) > 0

        # Check specific sections
        paths = [path for path, _ in sections]
        assert ["Test Policy", "1. Overview"] in paths
        assert ["Test Policy", "2. Details", "2.1 Subsection"] in paths

    def test_generate_document_id(self):
        """Test document ID generation is idempotent."""
        content1 = "Test content"
        content2 = "Test content"
        content3 = "Different content"

        id1 = generate_document_id(content1)
        id2 = generate_document_id(content2)
        id3 = generate_document_id(content3)

        assert id1 == id2  # Same content = same ID
        assert id1 != id3  # Different content = different ID
        assert len(id1) == 16  # Check length

    def test_generate_chunk_id(self):
        """Test chunk ID generation is idempotent."""
        doc_id = "doc123"
        section = "Section 1"
        text = "Test text"

        id1 = generate_chunk_id(doc_id, section, text)
        id2 = generate_chunk_id(doc_id, section, text)
        id3 = generate_chunk_id(doc_id, section, "Different text")

        assert id1 == id2  # Same inputs = same ID
        assert id1 != id3  # Different text = different ID
        assert len(id1) == 16

    def test_count_tokens(self):
        """Test token counting estimation."""
        text1 = "a" * 100
        text2 = "a" * 400

        count1 = count_tokens(text1)
        count2 = count_tokens(text2)

        assert count1 == 25  # 100 / 4
        assert count2 == 100  # 400 / 4

    def test_create_chunk(self):
        """Test PolicyChunk creation."""
        chunk = create_policy_chunk(
            document_id="doc123",
            metadata={
                "document_name": "Test Doc",
                "topic": "Test",
                "country": "CH",
                "active": True,
                "last_modified": "2024-01-01",
            },
            section_path=["Header1", "Header2"],
            section_path_str="Header1 > Header2",
            text="Test chunk text",
            chunk_index=0,
        )

        assert isinstance(chunk, PolicyChunk)
        assert chunk.document_id == "doc123"
        assert chunk.document_name == "Test Doc"
        assert chunk.section_path == ["Header1", "Header2"]
        assert chunk.section_path_str == "Header1 > Header2"
        assert chunk.text == "Test chunk text"
        assert chunk.chunk_index == 0
        assert chunk.country == "CH"
        assert chunk.active is True

        # Check text_indexed has prepended section path
        assert "Header1 > Header2" in chunk.text_indexed
        assert "Test chunk text" in chunk.text_indexed

    def test_load_document(self, pipeline, sample_document, tmp_path):
        """Test loading document from JSON file."""
        # Create temp JSON file
        test_file = tmp_path / "test_policy.json"
        with open(test_file, "w") as f:
            json.dump(sample_document, f)

        # Load it
        loaded = pipeline.load_document(test_file)

        assert loaded == sample_document
        assert loaded["metadata"]["document_name"] == "Test Policy"

    def test_create_chunks_from_document(self, pipeline, sample_document):
        """Test creating chunks from a document."""
        chunks = pipeline.create_chunks(sample_document)

        assert len(chunks) > 0
        assert all(isinstance(chunk, PolicyChunk) for chunk in chunks)

        # Check metadata is propagated
        for chunk in chunks:
            assert chunk.document_name == "Test Policy"
            assert chunk.topic == "Testing"
            assert chunk.country == "CH"
            assert chunk.active is True

        # Check chunk indices are sequential
        indices = [chunk.chunk_index for chunk in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunks_have_section_paths(self, pipeline, sample_document):
        """Test that chunks have proper section paths."""
        chunks = pipeline.create_chunks(sample_document)

        # All chunks should have section paths
        for chunk in chunks:
            assert len(chunk.section_path) > 0
            assert chunk.section_path_str != ""
            assert " > " in chunk.section_path_str or len(chunk.section_path) == 1


class TestPolicyChunkDataclass:
    """Tests for PolicyChunk dataclass."""

    def test_policy_chunk_creation(self):
        """Test creating a PolicyChunk."""
        chunk = PolicyChunk(
            document_id="doc123",
            document_name="Test Doc",
            section_path=["Section 1"],
            section_path_str="Section 1",
            chunk_id="chunk123",
            chunk_index=0,
            text="Test text",
            text_indexed="Section 1\n\nTest text",
            topic="Test",
            country="CH",
            active=True,
            last_modified="2024-01-01",
        )

        assert chunk.document_id == "doc123"
        assert chunk.chunk_index == 0
        assert chunk.active is True

    def test_policy_chunk_asdict(self):
        """Test converting PolicyChunk to dict."""
        from dataclasses import asdict

        chunk = PolicyChunk(
            document_id="doc123",
            document_name="Test Doc",
            section_path=["Section 1"],
            section_path_str="Section 1",
            chunk_id="chunk123",
            chunk_index=0,
            text="Test text",
            text_indexed="Section 1\n\nTest text",
            topic="Test",
            country="CH",
            active=True,
            last_modified="2024-01-01",
        )

        chunk_dict = asdict(chunk)
        assert isinstance(chunk_dict, dict)
        assert chunk_dict["document_id"] == "doc123"
        assert chunk_dict["section_path"] == ["Section 1"]


class TestHeaderParsing:
    """Tests for markdown header parsing."""

    def test_parse_simple_headers(self):
        """Test parsing simple markdown headers."""
        content = """# Header 1
Content 1

## Header 2
Content 2"""

        sections = split_by_headers(content)

        assert len(sections) == 2
        paths = [path for path, _ in sections]
        assert ["Header 1"] in paths
        assert ["Header 1", "Header 2"] in paths

    def test_parse_nested_headers(self):
        """Test parsing nested headers."""
        content = """# H1
## H2
### H3
Content"""

        sections = split_by_headers(content)

        paths = [path for path, _ in sections]
        assert ["H1", "H2", "H3"] in paths

    def test_header_level_changes(self):
        """Test handling header level changes."""
        content = """# H1
Content 1
### H3
Content 2
## H2
Content 3"""

        sections = split_by_headers(content)

        # Should handle jumps in header levels
        assert len(sections) >= 2
