import logging
from typing import Dict, Any

from chat_rag.agent.highlighting import SourceHighlighter, format_highlighted_sources
from ..state import AgentState

logger = logging.getLogger(__name__)


class HighlightingNode:
    """Handles source highlighting."""

    def __init__(self, highlighter: SourceHighlighter):
        self.highlighter = highlighter

    async def highlight_sources(self, state: AgentState) -> Dict:
        """Add source highlighting to response."""
        if not self.highlighter:
            return {}

        retrieval_state = state.get("retrieval", {})
        chunks = retrieval_state.get("chunks", [])
        if not chunks:
            return {}

        query = retrieval_state.get("resolved_query", "")
        answer = state.get("final_response", "")

        try:
            # Reconstruct chunks objects if needed or pass list of dicts if highlighter supports it
            # The original code passed 'chunks' which wait were retrieved objects, not just dicts.
            # But the state now stores dicts (json serializable).
            # SourceHighlighter.highlight_sync expects objects with 'text' attribute usually, or maybe dicts.
            # Let's check format_highlighted_sources and highlight_sync usage.
            # Original code: chunks = state.get("_chunks", []) which were objects.
            # Here I stored dicts in state["retrieval"]["chunks"].
            # I might need to adapt chunks to what highlighter expects.
            # Assuming highlighter can handle dicts or objects with attribute access.
            # If not, I can create a simple wrapper.

            # Helper to make dict accessible as object
            class DictObj:
                def __init__(self, d):
                    self.__dict__ = d

            chunk_objs = [DictObj(c) for c in chunks]

            highlighted = self.highlighter.highlight_sync(query, answer, chunk_objs)
            formatted = format_highlighted_sources(highlighted)

            if formatted:
                return {"highlighted_sources": formatted}

        except Exception as e:
            logger.error(f"Highlighting failed: {e}")

        return {}
