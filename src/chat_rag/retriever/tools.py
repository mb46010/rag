"""
Tools for integrating the retriever with LangChain/LlamaIndex agents.
"""

import logging
from typing import Optional, Callable
from .hybrid import EnhancedHybridRetriever

logger = logging.getLogger(__name__)


def create_retriever_tool(retriever: EnhancedHybridRetriever) -> Callable:
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


def create_langchain_tool(retriever: EnhancedHybridRetriever):
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

        return "\n\n".join(
            [f"Source: {c.document_name} (Section: {c.section_path_str})\n{c.text}" for c in chunks]
        )

    return search_policies
