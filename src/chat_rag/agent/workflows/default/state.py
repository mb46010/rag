from typing import TypedDict, Annotated, Optional, List, Dict
from enum import Enum
from langgraph.graph.message import add_messages


class QueryIntent(Enum):
    """Classified query intents for routing."""

    POLICY_ONLY = "policy_only"  # General policy questions
    PERSONAL_ONLY = "personal_only"  # User's specific data
    HYBRID = "hybrid"  # Needs both personal data AND policies
    CHITCHAT = "chitchat"  # Greetings, casual conversation
    OUT_OF_SCOPE = "out_of_scope"  # Not HR-related


class AgentState(TypedDict):
    """State that flows through the workflow graph."""

    messages: Annotated[list, add_messages]
    user_email: str
    intent: Optional[str]
    employee_profile: Optional[Dict]
    time_off_balance: Optional[Dict]
    policy_results: Optional[List[Dict]]
    error: Optional[str]
    final_response: Optional[str]
