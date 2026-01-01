"""
Tools for integrating the retriever with LangChain/LlamaIndex agents.
"""

import logging
from typing import Optional, Callable, List, Dict, TYPE_CHECKING
import weaviate
from weaviate.classes.query import Filter
from .models import PolicyChunk

if TYPE_CHECKING:
    from .hybrid import EnhancedHybridRetriever

logger = logging.getLogger(__name__)


def serve_chunks(client: weaviate.WeaviateClient, chunks: List[PolicyChunk]) -> None:
    """
    Fetch and serve previous/next chunks for context.

    This function batch-fetches adjacent chunks to avoid N+1 queries and
    populates the neighbor fields in the provided chunks.

    Args:
        client: Active Weaviate client
        chunks: List of PolicyChunk objects to populate
    """
    if not chunks:
        return

    # Identify unique (document_id, chunk_index) pairs needed
    adjacent_requests = set()
    collection_name = "PolicyChunk"  # Default assumption, ideally passed or inspected

    # Try to deduce collection name from client schema or config if available,
    # but for now we'll assume the standard name as per config logic.
    # Ideally should be passed in, but client doesn't store "current collection".

    for chunk in chunks:
        # Check if we need previous chunk
        if chunk.chunk_index > 0:
            adjacent_requests.add((chunk.document_id, chunk.chunk_index - 1))

        # Check if we need next chunk (we don't know max index easily, so we just try)
        adjacent_requests.add((chunk.document_id, chunk.chunk_index + 1))

    if not adjacent_requests:
        return

    try:
        collection = client.collections.get(collection_name)

        # Build OR filter for all adjacent chunks
        filter_expr = None
        for doc_id, idx in adjacent_requests:
            f = Filter.by_property("document_id").equal(doc_id) & Filter.by_property("chunk_index").equal(idx)
            filter_expr = f if filter_expr is None else (filter_expr | f)

        if filter_expr is None:
            return

        # Fetch objects (active & inactive to ensure we don't break chains if possible,
        # though usually we only index active ones. Let's assume we want whatever is there.)
        response = collection.query.fetch_objects(filters=filter_expr, limit=len(adjacent_requests))

        # Build lookup map: (doc_id, index) -> {text, chunk_id}
        context_map = {}
        for obj in response.objects:
            props = obj.properties
            key = (props.get("document_id"), props.get("chunk_index"))
            context_map[key] = {
                "text": props.get("text"),
                "chunk_id": props.get("chunk_id"),
                "section_path": props.get("section_path", []) or [props.get("section_path_str", "")],
                "active": props.get("active", True),  # Should almost always be true if retrievable
            }

        # Populate chunks
        for chunk in chunks:
            # Previous Chunk
            prev_key = (chunk.document_id, chunk.chunk_index - 1)
            prev_data = context_map.get(prev_key)

            if prev_data:
                # Optional: Check section hierarchy strictness if needed?
                # For now, we trust document_id continuity.
                chunk.previous_chunk = prev_data["text"]
                chunk.previous_chunk_id = prev_data["chunk_id"]

            # Next Chunk
            next_key = (chunk.document_id, chunk.chunk_index + 1)
            next_data = context_map.get(next_key)

            if next_data:
                chunk.next_chunk = next_data["text"]
                chunk.next_chunk_id = next_data["chunk_id"]

    except Exception as e:
        logger.warning(f"Error in serve_chunks: {e}")


def create_retriever_tool(retriever: "EnhancedHybridRetriever") -> Callable:
    """
    Create a tool function for LangChain agent.

    Args:
        retriever: EnhancedHybridRetriever instance

    Returns:
        Callable tool function compatible with LangChain
    """

    def search_policies(query: str, country: Optional[str] = None) -> str:
        """
        Search HR policies and retrieve relevant chunks.

        Args:
            query: Search query describing what policy information is needed
            country: Optional country filter (e.g., "CH", "IT")

        Returns:
            Formatted string with relevant policy chunks and citations
        """
        logger.info(f"search_policies called with query='{query}', country={country}")

        # Build filters
        filters = {"active": True}
        if country:
            filters["country"] = country

        # Retrieve chunks
        chunks = retriever.retrieve(query=query, top_k=3, filters=filters)

        if not chunks:
            return (
                "No relevant policy information found. "
                "Please rephrase your query or check if policies exist for the specified criteria."
            )

        # Format response with confidence indicators
        result = "Retrieved Policy Information:\n\n"
        for i, chunk in enumerate(chunks, 1):
            confidence_emoji = {"high": "✓", "medium": "~", "low": "?"}.get(chunk.confidence_level, "?")

            result += f"[{i}] {chunk.document_name} - {chunk.section_path_str} [{confidence_emoji}]\n"
            result += f"    Country: {chunk.country} | Last Modified: {chunk.last_modified}\n"

            # Include context if available
            if chunk.previous_chunk:
                result += f"    [Previous context]: {chunk.previous_chunk[:100]}...\n"

            result += f"    {chunk.text}\n"

            if chunk.next_chunk:
                result += f"    [Following context]: {chunk.next_chunk[:100]}...\n"

            result += f"    (Score: {chunk.score:.3f}, Confidence: {chunk.confidence_level})\n\n"

        return result

    return search_policies


def create_langchain_tool(retriever: "EnhancedHybridRetriever"):
    """
    Create a LangChain Tool object for the retriever.

    Returns:
        LangChain Tool object
    """
    from langchain.tools import tool

    @tool
    def search_policies(query: str, country: Optional[str] = None) -> str:
        """Search HR policies and retrieve relevant information.

        Use this tool to find policy information. Use the country filter
        when you know the employee's country for more relevant results.

        Args:
            query: What policy information to search for
            country: Optional country code (e.g., "CH", "IT")
        """
        filters = {"active": True}
        if country:
            filters["country"] = country

        chunks = retriever.retrieve(query=query, top_k=3, filters=filters)

        if not chunks:
            return "No relevant policy information found."

        return "\n\n".join([f"Source: {c.document_name} (Section: {c.section_path_str})\n{c.text}" for c in chunks])

    return search_policies
