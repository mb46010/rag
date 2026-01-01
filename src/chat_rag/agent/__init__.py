"""
Agent module for HR Assistant.

Provides both the new workflow-based agent and backwards-compatible
legacy agent interface.
"""

from .workflow import HRWorkflowAgent, HRAssistantAgent, QueryIntent, AgentState
from .config import AgentConfig
from .prompt import SYSTEM_PROMPT, SYNTHESIS_SYSTEM_PROMPT

__all__ = [
    "HRWorkflowAgent",
    "HRAssistantAgent",  # Backwards compatibility alias
    "QueryIntent",
    "AgentState",
    "AgentConfig",
    "SYSTEM_PROMPT",
    "SYNTHESIS_SYSTEM_PROMPT",
]
