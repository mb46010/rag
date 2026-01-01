from typing import TypedDict, Annotated, Optional, List, Dict
from langgraph.graph.message import add_messages


class UserContext(TypedDict):
    """User context information."""

    email: str
    profile: Optional[Dict]
    pto_balance: Optional[Dict]


class RetrievalContext(TypedDict):
    """Retrieval related state."""

    original_query: str
    resolved_query: str
    chunks: List[Dict]
    hyde_used: bool


class ResponseContext(TypedDict):
    """Response generation state."""

    intent: Optional[str]
    needs_clarification: bool
    clarifying_question: Optional[str]


class AgentState(TypedDict):
    """Cleaner state with logical groupings."""

    messages: Annotated[list, add_messages]

    # Grouped contexts
    user: UserContext
    retrieval: RetrievalContext
    response: ResponseContext

    # Final output
    final_response: Optional[str]
    highlighted_sources: Optional[str]
    error: Optional[str]

    # Tracking
    query_id: str
