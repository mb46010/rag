"""Agent module for HR Assistant.

Provides:
- HRWorkflowAgent: Base workflow agent
- EnhancedHRWorkflowAgent: Full-featured agent with HyDE, clarification, etc.
- Supporting modules: clarification, highlighting, memory
"""

from .workflow import HRWorkflowAgent, HRAssistantAgent, QueryIntent, AgentState
from .enhanced_workflow import EnhancedHRWorkflowAgent, EnhancedAgentState
from .config import AgentConfig
from .prompt import SYSTEM_PROMPT, SYNTHESIS_SYSTEM_PROMPT
from .clarification import ClarificationDetector, ClarificationResult, AmbiguityType
from .highlighting import SourceHighlighter, HighlightedSource, format_highlighted_sources
from .memory import SessionMemory, ConversationTurn, SessionContext

__all__ = [
    # Agents
    "HRWorkflowAgent",
    "HRAssistantAgent",
    "EnhancedHRWorkflowAgent",
    # State
    "QueryIntent",
    "AgentState",
    "EnhancedAgentState",
    # Config
    "AgentConfig",
    # Prompts
    "SYSTEM_PROMPT",
    "SYNTHESIS_SYSTEM_PROMPT",
    # Clarification
    "ClarificationDetector",
    "ClarificationResult",
    "AmbiguityType",
    # Highlighting
    "SourceHighlighter",
    "HighlightedSource",
    "format_highlighted_sources",
    # Memory
    "SessionMemory",
    "ConversationTurn",
    "SessionContext",
    "get_agent",
]


def get_agent(
    user_email: str, retriever=None, config: AgentConfig = None
) -> HRWorkflowAgent | EnhancedHRWorkflowAgent:
    """Factory function to get the appropriate agent based on configuration.

    Args:
        user_email: Email of the user
        retriever: Optional retriever instance
        config: Optional agent configuration. If None, default AgentConfig is used.

    Returns:
        An instance of HRWorkflowAgent or EnhancedHRWorkflowAgent
    """
    if config is None:
        config = AgentConfig()

    if config.use_enhanced_workflow:
        return EnhancedHRWorkflowAgent(
            user_email=user_email,
            retriever=retriever,
            config=config,
            enable_hyde=config.enable_hyde,
            enable_clarification=config.enable_clarification,
            enable_highlighting=config.enable_highlighting,
            enable_memory=config.enable_memory,
            enable_cost_tracking=config.enable_cost_tracking,
        )
    else:
        return HRWorkflowAgent(user_email=user_email, retriever=retriever, config=config)
