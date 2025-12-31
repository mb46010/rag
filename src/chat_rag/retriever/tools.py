import logging
from typing import Optional
from .hybrid import HybridRetriever

logger = logging.getLogger(__name__)


def create_retriever_tool(retriever: HybridRetriever) -> callable:
    """Create a tool function for LangChain agent.

    Args:
        retriever: HybridRetriever instance

    Returns:
        Callable tool function
    """

    def search_policies(query: str, country: Optional[str] = None) -> str:
        """Search HR policies and retrieve relevant chunks.

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
            return "No relevant policy information found. Please rephrase your query or check if policies exist for the specified criteria."

        # Format response
        result = "Retrieved Policy Information:\n\n"
        for i, chunk in enumerate(chunks, 1):
            result += f"[{i}] {chunk.document_name} - {chunk.section_path_str}\n"
            result += f"    Country: {chunk.country} | Last Modified: {chunk.last_modified}\n"

            # Include context if available
            if chunk.previous_chunk:
                result += f"    [Previous context]: {chunk.previous_chunk[:100]}...\n"

            result += f"    {chunk.text}\n"

            if chunk.next_chunk:
                result += f"    [Following context]: {chunk.next_chunk[:100]}...\n"

            result += f"    (Relevance score: {chunk.score:.3f})\n\n"

        return result

    return search_policies
