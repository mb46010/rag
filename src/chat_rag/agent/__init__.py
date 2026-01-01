"""
Agent module for HR Assistant.

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
]
