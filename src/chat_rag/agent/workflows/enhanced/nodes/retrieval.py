import logging
from typing import Dict, List, Any, Optional

from chat_rag.retriever import EnhancedHybridRetriever
from ..state import AgentState

logger = logging.getLogger(__name__)


class RetrievalNode:
    """Handles policy retrieval."""

    def __init__(self, retriever: EnhancedHybridRetriever, enable_hyde: bool = True):
        self.retriever = retriever
        self.enable_hyde = enable_hyde

    async def search_policies(self, state: AgentState) -> Dict:
        """Search for policies based on resolved query."""
        query = state["retrieval"].get("resolved_query") or state["retrieval"].get("original_query")

        # Build filters from profile
        filters = self._build_filters(state)

        # Execute retrieval
        chunks = []
        hyde_used = False

        try:
            if self.enable_hyde:
                # Check if retriever supports async
                if hasattr(self.retriever, "aretrieve"):
                    chunks = await self.retriever.aretrieve(query, filters=filters, top_k=3)
                else:
                    chunks = self.retriever.retrieve(query, filters=filters, top_k=3)

                # Check if HyDE was actually used (depends on implementation)
                hyde_used = any(getattr(c, "hyde_enhanced", False) for c in chunks)
            else:
                if hasattr(self.retriever, "aretrieve"):
                    chunks = await self.retriever.aretrieve(query, filters=filters, top_k=3)
                else:
                    chunks = self.retriever.retrieve(query, filters=filters, top_k=3)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {"error": f"Retrieval failed: {str(e)}"}

        formatted_chunks = [self._format_chunk(c) for c in chunks]

        return {
            "retrieval": {
                **state.get("retrieval", {}),
                "chunks": formatted_chunks,
                "hyde_used": hyde_used,
            }
        }

    def _build_filters(self, state: AgentState) -> Dict:
        filters = {"active": True}
        # Safely access active user profile country
        user_info = state.get("user", {})
        profile = user_info.get("profile")

        if profile and isinstance(profile, dict):
            country = profile.get("country")
            if country:
                filters["country"] = country
        return filters

    def _format_chunk(self, chunk) -> Dict:
        return {
            "text": chunk.text,
            "source": f"{chunk.document_name} > {chunk.section_path_str}",
            "confidence": chunk.confidence_level,
            "score": chunk.score,
            "chunk_id": getattr(chunk, "chunk_id", None),
        }
