"""
Retriever module for HR policy document search.

Provides hybrid search combining BM25 and vector similarity
with optional reranking and context windows.
"""

from .models import PolicyChunk, QueryType, RetrievalResult
from .config import RetrievalConfig
from .hybrid import EnhancedHybridRetriever, HybridRetriever
from .reranker import Reranker
from .tools import create_retriever_tool, create_langchain_tool

__all__ = [
    "PolicyChunk",
    "QueryType",
    "RetrievalResult",
    "RetrievalConfig",
    "EnhancedHybridRetriever",
    "HybridRetriever",
    "Reranker",
    "create_retriever_tool",
    "create_langchain_tool",
]
