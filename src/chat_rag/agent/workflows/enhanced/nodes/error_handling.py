import logging
from typing import Dict, Any
from ..state import AgentState

logger = logging.getLogger(__name__)


async def handle_error(state: AgentState) -> Dict:
    """Graceful degradation when errors occur."""
    # Since we can't easily access the global error from just "error" key if it was set in a nested dict
    # We rely on nodes setting the top-level 'error' key if they want to trigger this,
    # OR we check nested errors.
    # Current design of nodes: return {"error": ...} often updating a specific field.
    # The graph conditional edge needs to check if error exists.

    error = state.get("error")

    # Try to give a helpful response
    msg = "I encountered an unexpected issue. Please try again or contact HR support."

    if error and "profile" in error.lower():
        msg = "I encountered an issue fetching your profile. I can still help with general policy questions but personal details might be unavailable."
    elif error:
        msg = f"I encountered an error: {error}. Please try again."

    return {
        "final_response": msg,
        "error": None,  # Clear error
    }
