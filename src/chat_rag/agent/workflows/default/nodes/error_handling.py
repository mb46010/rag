import logging
from typing import Dict, Any
from ..state import AgentState

logger = logging.getLogger(__name__)


async def handle_error(state: AgentState) -> Dict:
    """Graceful degradation when errors occur."""
    error = state.get("error")

    msg = "I encountered an unexpected issue. Please try again or contact HR support."
    if error:
        msg = f"I encountered an error: {error}. Please try again."

    return {"final_response": msg, "error": None}
