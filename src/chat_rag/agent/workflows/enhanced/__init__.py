from .workflow import EnhancedHRWorkflowAgent
from .state import AgentState

# Alias for backward compatibility
EnhancedAgentState = AgentState

__all__ = ["EnhancedHRWorkflowAgent", "AgentState", "EnhancedAgentState"]
