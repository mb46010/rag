from typing import TypedDict, Annotated, Optional, List, Dict
from langgraph.graph.message import add_messages


class EnhancedAgentState(TypedDict):
    """State with additional fields for new features."""

    messages: Annotated[list, add_messages]
    user_email: str
    query_id: str

    # Original and resolved query
    original_query: str
    resolved_query: str

    # Intent and routing
    intent: Optional[str]
    needs_clarification: bool
    clarifying_question: Optional[str]

    # Data gathered
    employee_profile: Optional[Dict]
    time_off_balance: Optional[Dict]
    policy_results: Optional[List[Dict]]

    # Enhanced features
    highlighted_sources: Optional[str]
    hyde_used: bool

    # Output
    final_response: Optional[str]
    error: Optional[str]

    # Internal usage for highlighting
    _chunks: Optional[List]
