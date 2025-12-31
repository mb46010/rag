import pytest
from unittest.mock import MagicMock, patch
from src.chat_rag.retriever.reranker import Reranker
from src.chat_rag.retriever.models import PolicyChunk


@pytest.fixture
def mock_cross_encoder():
    with patch("src.chat_rag.retriever.reranker.CrossEncoder") as mock:
        yield mock


def test_reranker_initialization(mock_cross_encoder):
    model_name = "test-model"
    reranker = Reranker(model_name=model_name)

    mock_cross_encoder.assert_called_once_with(model_name, device="cpu")
    assert reranker.model_name == model_name


def test_reranker_rerank(mock_cross_encoder):
    reranker = Reranker()
    mock_model = mock_cross_encoder.return_value

    # Mock return scores: 0.1 for first chunk, 0.9 for second
    mock_model.predict.return_value = [0.1, 0.9]

    chunk1 = PolicyChunk(
        text="chunk1",
        document_id="1",
        document_name="doc1",
        section_path_str="",
        chunk_id="c1",
        chunk_index=0,
        topic="",
        country="",
        active=True,
        last_modified="",
        score=0.5,
    )
    chunk2 = PolicyChunk(
        text="chunk2",
        document_id="2",
        document_name="doc2",
        section_path_str="",
        chunk_id="c2",
        chunk_index=0,
        topic="",
        country="",
        active=True,
        last_modified="",
        score=0.4,
    )

    chunks = [chunk1, chunk2]
    query = "test query"

    reranked = reranker.rerank(query, chunks)

    # Verify predict was called with correct pairs
    mock_model.predict.assert_called_once_with([[query, "chunk1"], [query, "chunk2"]])

    # Verify chunks are sorted by new score (descending)
    assert len(reranked) == 2
    assert reranked[0] == chunk2
    assert reranked[1] == chunk1
    assert reranked[0].score == 0.9
    assert reranked[1].score == 0.1


def test_reranker_rerank_top_k(mock_cross_encoder):
    reranker = Reranker()
    mock_model = mock_cross_encoder.return_value

    # 3 chunks, scores: 0.1, 0.9, 0.5
    mock_model.predict.return_value = [0.1, 0.9, 0.5]

    chunks = [
        PolicyChunk(
            text=f"chunk{i}",
            document_id=str(i),
            document_name="",
            section_path_str="",
            chunk_id="",
            chunk_index=0,
            topic="",
            country="",
            active=True,
            last_modified="",
            score=0.0,
        )
        for i in range(3)
    ]

    reranked = reranker.rerank("query", chunks, top_k=1)

    assert len(reranked) == 1
    assert reranked[0].text == "chunk1"  # The one with score 0.9
    assert reranked[0].score == 0.9


def test_reranker_empty_chunks(mock_cross_encoder):
    reranker = Reranker()
    assert reranker.rerank("query", []) == []
