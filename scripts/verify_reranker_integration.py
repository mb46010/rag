import logging
import sys
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_integration():
    logger.info("Verifying HybridRetriever integration with Reranker...")

    # Mock Weaviate and external dependencies
    with (
        patch("src.chat_rag.retriever.hybrid.weaviate"),
        patch("src.chat_rag.retriever.hybrid.OpenAIEmbedding"),
        patch("src.chat_rag.retriever.hybrid.WeaviateVectorStore"),
        patch("src.chat_rag.retriever.hybrid.Reranker") as MockReranker,
    ):
        from src.chat_rag.retriever.hybrid import HybridRetriever

        # Test initialization with reranker
        logger.info("Testing initialization with reranker_model...")
        retriever = HybridRetriever(reranker_model="test-model")

        if retriever.reranker is not None:
            logger.info("SUCCESS: Reranker initialized.")
        else:
            logger.error("FAILURE: Reranker not initialized.")
            sys.exit(1)

        MockReranker.assert_called_with(model_name="test-model")

        # Test initialization without reranker
        logger.info("Testing initialization without reranker_model...")
        retriever_no_rerank = HybridRetriever(reranker_model=None)

        if retriever_no_rerank.reranker is None:
            logger.info("SUCCESS: Reranker correctly not initialized.")
        else:
            logger.error("FAILURE: Reranker initialized when it shouldn't be.")
            sys.exit(1)


if __name__ == "__main__":
    verify_integration()
