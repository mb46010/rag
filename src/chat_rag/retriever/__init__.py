"""
Retriever module for HR policy document search.

Provides hybrid search combining BM25 and vector similarity
with optional reranking, context windows, and HyDE.
"""

from .models import PolicyChunk, QueryType, RetrievalResult
from .config import RetrievalConfig
from .hybrid import EnhancedHybridRetriever, HybridRetriever
from .reranker import Reranker
from .tools import create_retriever_tool, create_langchain_tool
from .hyde import HyDERetriever

__all__ = [
    # Models
    "PolicyChunk",
    "QueryType",
    "RetrievalResult",
    # Config
    "RetrievalConfig",
    # Retrievers
    "EnhancedHybridRetriever",
    "HybridRetriever",
    "HyDERetriever",
    # Reranking
    "Reranker",
    # Tools
    "create_retriever_tool",
    "create_langchain_tool",
]
