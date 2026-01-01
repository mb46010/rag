import logging
import asyncio
from typing import Dict, Any, List

from chat_rag.retriever import EnhancedHybridRetriever

from ..state import AgentState
# We will import MCPNode later or inject dependencies to avoid circular imports
# or use a helper for parallel fetching if needed.
# But here parallel fetch is logic combining two sources.

logger = logging.getLogger(__name__)


class RetrievalNode:
    """Handles policy retrieval."""

    def __init__(self, retriever: EnhancedHybridRetriever):
        self.retriever = retriever

    async def search_policies(self, state: AgentState) -> Dict:
        """Search policy documents."""
        query = state["messages"][-1].content

        filters = {"active": True}
        user_state = state.get("user", {})
        if user_state.get("profile") and isinstance(user_state["profile"], dict):
            country = user_state["profile"].get("country")
            if country:
                filters["country"] = country

        try:
            # Use aretrieve for consistency and HyDE support
            if hasattr(self.retriever, "aretrieve"):
                chunks = await self.retriever.aretrieve(query, filters=filters, top_k=3)
            else:
                chunks = self.retriever.retrieve(query, filters=filters, top_k=3)

            policy_results = [
                {
                    "text": c.text,
                    "document_name": c.document_name,
                    "section_path_str": c.section_path_str,
                    "source": f"{c.document_name} > {c.section_path_str}",
                    "confidence": c.confidence_level,
                    "score": c.score,
                }
                for c in chunks
            ]

            return {"retrieval": {"policy_results": policy_results}}
        except Exception as e:
            logger.error(f"Policy search failed: {e}")
            return {"error": f"Policy search failed: {str(e)}"}
