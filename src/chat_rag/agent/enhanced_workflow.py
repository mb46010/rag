"""
Enhanced LangGraph Workflow Agent with production features.

Features:
- HyDE for improved retrieval
- Clarifying questions for ambiguous queries
- Source highlighting for citations
- Multi-turn session memory
- Cost tracking

Refactored to src/chat_rag/agent/workflows/enhanced/
"""

from .workflows.enhanced import EnhancedHRWorkflowAgent, EnhancedAgentState

__all__ = ["EnhancedHRWorkflowAgent", "EnhancedAgentState"]
