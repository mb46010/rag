"""
LangGraph Workflow Agent for HR Assistant.

This replaces the ReAct agent with a structured workflow that:
1. Classifies query intent
2. Routes to appropriate data gathering
3. Executes tools in parallel where possible
4. Synthesizes response with citations

Benefits over ReAct:
- Fewer LLM calls (2-3 vs 4-5)
- More predictable behavior
- Better error handling
- Parallel execution support

Refactored to src/chat_rag/agent/workflows/default/
"""

from .workflows.default import HRWorkflowAgent, AgentState, QueryIntent

# Keep the old class name for backwards compatibility
HRAssistantAgent = HRWorkflowAgent

__all__ = ["HRWorkflowAgent", "AgentState", "QueryIntent", "HRAssistantAgent"]
