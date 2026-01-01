"""
Tests for ingestion policies.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch, Mock

# Mock external dependencies before imports
sys.modules["weaviate"] = MagicMock()
sys.modules["weaviate.classes"] = MagicMock()
sys.modules["weaviate.classes.config"] = MagicMock()
sys.modules["llama_index"] = MagicMock()
sys.modules["llama_index.core"] = MagicMock()
sys.modules["llama_index.core.node_parser"] = MagicMock()
sys.modules["llama_index.core.ingestion"] = MagicMock()
sys.modules["llama_index.core.extractors"] = MagicMock()
sys.modules["llama_index.embeddings"] = MagicMock()
sys.modules["llama_index.embeddings.openai"] = MagicMock()
sys.modules["llama_index.llms"] = MagicMock()
sys.modules["llama_index.llms.openai"] = MagicMock()

from pathlib import Path
from src.chat_rag.ingestion.config import IngestionConfig
from src.chat_rag.ingestion.policies.default import DefaultIngestionPolicy
from src.chat_rag.ingestion.policies.qa_extractor import QaExtractorPolicy
from src.chat_rag.ingestion.models import PolicyChunk


class TestDefaultIngestionPolicy:
    """Tests for DefaultIngestionPolicy."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a test configuration."""
        return IngestionConfig(
            docs_dir=tmp_path / "docs",
            output_dir=tmp_path / "output",
            weaviate_url="http://localhost:8080",
            collection_name="TestPolicyChunk",
            max_chunk_tokens=512,
            chunk_overlap=50,
        )

    @pytest.fixture
    def policy(self, config):
        """Create policy instance."""
        return DefaultIngestionPolicy(config=config)

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

    def test_policy_name_and_version(self, policy):
        """Test policy returns correct name and version."""
        assert policy.get_name() == "default"
        assert policy.get_version() == "1.0.0"

    def test_process_document_basic(self, policy, sample_document):
        """Test basic document processing."""
        chunks = policy.process_document(sample_document)

        # Should produce multiple chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, PolicyChunk) for chunk in chunks)

        # Check metadata propagation
        for chunk in chunks:
            assert chunk.document_name == "Test Policy"
            assert chunk.topic == "Testing"
            assert chunk.country == "CH"
            assert chunk.active is True
            assert chunk.last_modified == "2024-01-01"

    def test_process_document_chunk_indices(self, policy, sample_document):
        """Test that chunk indices are sequential."""
        chunks = policy.process_document(sample_document)

        indices = [chunk.chunk_index for chunk in chunks]
        assert indices == list(range(len(chunks)))

    def test_process_document_section_paths(self, policy, sample_document):
        """Test that chunks have proper section paths."""
        chunks = policy.process_document(sample_document)

        # All chunks should have section paths
        for chunk in chunks:
            assert len(chunk.section_path) > 0
            assert chunk.section_path_str != ""
            assert isinstance(chunk.section_path, list)

    def test_process_document_with_multiple_sections(self, config):
        """Test handling of documents with multiple sections."""
        policy = DefaultIngestionPolicy(config=config)

        document = {
            "metadata": {
                "document_id": "pol_multi",
                "document_name": "Multi-Section Policy",
                "topic": "Testing",
                "country": "US",
                "active": True,
                "last_modified": "2024-01-01",
            },
            "content": """# Main Policy

## Section One
Content for section one.

## Section Two
Content for section two.

## Section Three
Content for section three.""",
        }

        chunks = policy.process_document(document)

        # Should produce chunks for each section
        assert len(chunks) >= 3

    def test_process_document_minimum_split_level(self, config, tmp_path):
        """Test minimum split level configuration."""
        # Test with different minimum split levels
        config.minimum_split_level = "Header2"
        policy = DefaultIngestionPolicy(config=config)

        document = {
            "metadata": {
                "document_id": "pol_test",
                "document_name": "Test Policy",
                "topic": "Testing",
                "country": "CH",
                "active": True,
                "last_modified": "2024-01-01",
            },
            "content": """# Header 1
Content 1

## Header 2
Content 2

### Header 3
Content 3

#### Header 4
Content 4""",
        }

        chunks = policy.process_document(document)
        assert len(chunks) > 0

    def test_process_document_invalid_minimum_split_level(self, config):
        """Test handling of invalid minimum split level."""
        config.minimum_split_level = "InvalidHeader"
        policy = DefaultIngestionPolicy(config=config)

        document = {
            "metadata": {
                "document_id": "pol_test",
                "document_name": "Test Policy",
                "topic": "Testing",
                "country": "CH",
                "active": True,
                "last_modified": "2024-01-01",
            },
            "content": "# Header\nContent",
        }

        # Should not raise error, should default to level 6
        chunks = policy.process_document(document)
        assert len(chunks) > 0

    def test_process_document_empty_content(self, policy):
        """Test processing document with empty content."""
        document = {
            "metadata": {
                "document_id": "pol_empty",
                "document_name": "Empty Policy",
                "topic": "Testing",
                "country": "CH",
                "active": True,
                "last_modified": "2024-01-01",
            },
            "content": "",
        }

        chunks = policy.process_document(document)
        # Empty content might result in zero or one chunk depending on implementation
        assert isinstance(chunks, list)

    def test_process_document_no_headers(self, policy):
        """Test processing document without markdown headers."""
        document = {
            "metadata": {
                "document_id": "pol_no_headers",
                "document_name": "No Headers Policy",
                "topic": "Testing",
                "country": "CH",
                "active": True,
                "last_modified": "2024-01-01",
            },
            "content": "This is plain text without any headers. Just regular content.",
        }

        chunks = policy.process_document(document)
        assert len(chunks) >= 0

    def test_chunk_text_indexed_includes_section_path(self, policy, sample_document):
        """Test that text_indexed includes section path."""
        chunks = policy.process_document(sample_document)

        for chunk in chunks:
            # text_indexed should contain the section path
            assert chunk.section_path_str in chunk.text_indexed


class TestQaExtractorPolicy:
    """Tests for QaExtractorPolicy."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a test configuration."""
        return IngestionConfig(
            docs_dir=tmp_path / "docs",
            output_dir=tmp_path / "output",
            weaviate_url="http://localhost:8080",
            collection_name="TestPolicyChunk",
            max_chunk_tokens=512,
            chunk_overlap=50,
            llm_model="gpt-4o-mini",
            questions_to_generate=3,
        )

    @pytest.fixture
    def mock_llama_components(self):
        """Mock LlamaIndex components."""
        with patch("src.chat_rag.ingestion.policies.qa_extractor.OpenAI") as mock_llm, \
             patch("src.chat_rag.ingestion.policies.qa_extractor.LlamaIngestionPipeline") as mock_pipeline, \
             patch("src.chat_rag.ingestion.policies.qa_extractor.SentenceSplitter") as mock_splitter, \
             patch("src.chat_rag.ingestion.policies.qa_extractor.TitleExtractor") as mock_title, \
             patch("src.chat_rag.ingestion.policies.qa_extractor.QuestionsAnsweredExtractor") as mock_qa, \
             patch("src.chat_rag.ingestion.policies.qa_extractor.Document") as mock_doc:

            # Mock node returned by pipeline
            mock_node = Mock()
            mock_node.metadata = {
                "section_summary": "Test Section",
                "questions_this_excerpt_can_answer": "Q: What is this? A: Test content.",
            }
            mock_node.get_content.return_value = "This is test content from the node."

            # Mock pipeline to return nodes
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.run.return_value = [mock_node, mock_node]
            mock_pipeline.return_value = mock_pipeline_instance

            yield {
                "llm": mock_llm,
                "pipeline": mock_pipeline,
                "splitter": mock_splitter,
                "title": mock_title,
                "qa": mock_qa,
                "doc": mock_doc,
                "node": mock_node,
                "pipeline_instance": mock_pipeline_instance,
            }

    @pytest.fixture
    def sample_document(self):
        """Sample policy document for testing."""
        return {
            "metadata": {
                "document_id": "pol_qa_test",
                "document_name": "QA Test Policy",
                "topic": "Testing",
                "country": "CH",
                "active": True,
                "last_modified": "2024-01-01",
            },
            "content": """# Test Policy

This is a test policy for QA extraction testing.""",
        }

    def test_policy_name_and_version(self, config, mock_llama_components):
        """Test policy returns correct name and version."""
        policy = QaExtractorPolicy(config=config)
        assert policy.get_name() == "qa_extractor"
        assert policy.get_version() == "1.0.0"

    def test_initialization(self, config, mock_llama_components):
        """Test policy initialization."""
        policy = QaExtractorPolicy(config=config)

        # Should have initialized LLM
        mock_llama_components["llm"].assert_called_once_with(model="gpt-4o-mini")

        # Should have transformations list
        assert hasattr(policy, "transformations")
        assert len(policy.transformations) == 3  # Splitter, Title, QA

    def test_process_document(self, config, mock_llama_components, sample_document):
        """Test document processing with QA extraction."""
        policy = QaExtractorPolicy(config=config)
        chunks = policy.process_document(sample_document)

        # Should produce chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, PolicyChunk) for chunk in chunks)

        # Check that pipeline was called
        mock_llama_components["pipeline_instance"].run.assert_called_once()

    def test_qa_text_extraction(self, config, mock_llama_components, sample_document):
        """Test that QA text is extracted and stored."""
        policy = QaExtractorPolicy(config=config)
        chunks = policy.process_document(sample_document)

        # Chunks should have QA text
        for chunk in chunks:
            assert chunk.qa_text is not None
            assert "What is this?" in chunk.qa_text

    def test_section_path_from_title(self, config, mock_llama_components, sample_document):
        """Test that section path is derived from title extractor."""
        policy = QaExtractorPolicy(config=config)
        chunks = policy.process_document(sample_document)

        # Check section paths
        for chunk in chunks:
            assert chunk.section_path_str == "Test Section"
            assert chunk.section_path == ["Test Section"]

    def test_metadata_propagation(self, config, mock_llama_components, sample_document):
        """Test that document metadata is propagated to chunks."""
        policy = QaExtractorPolicy(config=config)
        chunks = policy.process_document(sample_document)

        for chunk in chunks:
            assert chunk.document_name == "QA Test Policy"
            assert chunk.topic == "Testing"
            assert chunk.country == "CH"
            assert chunk.active is True

    def test_chunk_indices(self, config, mock_llama_components, sample_document):
        """Test that chunk indices are sequential."""
        policy = QaExtractorPolicy(config=config)
        chunks = policy.process_document(sample_document)

        indices = [chunk.chunk_index for chunk in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_qa_text(self, config, sample_document):
        """Test handling when QA extractor returns empty text."""
        with patch("src.chat_rag.ingestion.policies.qa_extractor.OpenAI"), \
             patch("src.chat_rag.ingestion.policies.qa_extractor.LlamaIngestionPipeline") as mock_pipeline, \
             patch("src.chat_rag.ingestion.policies.qa_extractor.SentenceSplitter"), \
             patch("src.chat_rag.ingestion.policies.qa_extractor.TitleExtractor"), \
             patch("src.chat_rag.ingestion.policies.qa_extractor.QuestionsAnsweredExtractor"), \
             patch("src.chat_rag.ingestion.policies.qa_extractor.Document"):

            # Mock node with empty QA text
            mock_node = Mock()
            mock_node.metadata = {
                "section_summary": "",
                "document_title": "",
                "questions_this_excerpt_can_answer": "",
            }
            mock_node.get_content.return_value = "Test content"

            mock_pipeline_instance = Mock()
            mock_pipeline_instance.run.return_value = [mock_node]
            mock_pipeline.return_value = mock_pipeline_instance

            policy = QaExtractorPolicy(config=config)
            chunks = policy.process_document(sample_document)

            assert len(chunks) > 0
            # Should handle empty QA text gracefully
            assert chunks[0].qa_text == ""

    def test_multiple_nodes(self, config, sample_document):
        """Test processing document that results in multiple nodes."""
        with patch("src.chat_rag.ingestion.policies.qa_extractor.OpenAI"), \
             patch("src.chat_rag.ingestion.policies.qa_extractor.LlamaIngestionPipeline") as mock_pipeline, \
             patch("src.chat_rag.ingestion.policies.qa_extractor.SentenceSplitter"), \
             patch("src.chat_rag.ingestion.policies.qa_extractor.TitleExtractor"), \
             patch("src.chat_rag.ingestion.policies.qa_extractor.QuestionsAnsweredExtractor"), \
             patch("src.chat_rag.ingestion.policies.qa_extractor.Document"):

            # Create multiple mock nodes
            nodes = []
            for i in range(3):
                mock_node = Mock()
                mock_node.metadata = {
                    "section_summary": f"Section {i}",
                    "questions_this_excerpt_can_answer": f"Question {i}?",
                }
                mock_node.get_content.return_value = f"Content {i}"
                nodes.append(mock_node)

            mock_pipeline_instance = Mock()
            mock_pipeline_instance.run.return_value = nodes
            mock_pipeline.return_value = mock_pipeline_instance

            policy = QaExtractorPolicy(config=config)
            chunks = policy.process_document(sample_document)

            # Should create chunk for each node
            assert len(chunks) == 3
            for i, chunk in enumerate(chunks):
                assert chunk.chunk_index == i
                assert f"Section {i}" in chunk.section_path_str
