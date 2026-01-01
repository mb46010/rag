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


class UserContext(TypedDict):
    """User related context."""

    email: str
    profile: Optional[Dict]
    pto_balance: Optional[Dict]


class RetrievalContext(TypedDict):
    """Retrieval related state."""

    policy_results: Optional[List[Dict]]


class ResponseContext(TypedDict):
    """Response generation state."""

    intent: Optional[str]


class AgentState(TypedDict):
    """State that flows through the workflow graph."""

    messages: Annotated[list, add_messages]

    # Grouped contexts
    user: UserContext
    retrieval: RetrievalContext
    response: ResponseContext

    # Output
    final_response: Optional[str]
    error: Optional[str]
