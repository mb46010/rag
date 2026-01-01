import logging
import asyncio
from typing import Dict, Any

from ..state import AgentState
from .mcp import MCPNode
from .retrieval import RetrievalNode

logger = logging.getLogger(__name__)


class ParallelFetchNode:
    """Handles parallel data fetching."""

    def __init__(self, mcp_node: MCPNode, retrieval_node: RetrievalNode, enable_parallel: bool = True):
        self.mcp_node = mcp_node
        self.retrieval_node = retrieval_node
        self.enable_parallel = enable_parallel
        self.user_email = mcp_node.user_email  # helper

    async def parallel_fetch(self, state: AgentState) -> Dict:
        """Fetch personal data and policies in parallel."""
        if not self.enable_parallel:
            # Sequential fallback
            # Note: This reproduces original sequential logic where we fetch profile first, then policies

            # 1. Fetch personal data (Profile + PTO)
            personal_update = await self.mcp_node.fetch_personal_data(state)

            # Merge into temp state for second step
            state_with_profile = {**state, "user": {**state.get("user", {}), **personal_update.get("user", {})}}

            # 2. Search policies (using profile from step 1)
            policy_update = await self.retrieval_node.search_policies(state_with_profile)

            return {"user": state_with_profile["user"], "retrieval": policy_update.get("retrieval", {})}

        # Parallel execution (optimized from original)
        # Original: Profile -> (PTO + Policies)

        # 1. Get profile first (needed for policy filters)
        try:
            profile_response = await self.mcp_node.call_tool("get_employee_profile_tool", {"email": self.user_email})
            # Robust extraction
            profile = profile_response
            if isinstance(profile, dict) and "employee_profile" in profile:
                inner = profile["employee_profile"]
                if isinstance(inner, dict) and "text" in inner:
                    profile = inner["text"]
                else:
                    profile = inner
        except Exception as e:
            logger.error(f"Parallel fetch initial profile failed: {e}")
            profile = {"error": str(e)}

        # Update state for next steps
        current_user_state = state.get("user", {})
        temp_state = {**state, "user": {**current_user_state, "profile": profile}}

        # 2. Parallel PTO + Policies
        async def fetch_pto():
            try:
                bal = await self.mcp_node.call_tool("get_time_off_balance_tool", {"email": self.user_email})
                return bal
            except Exception as e:
                return {"error": str(e)}

        async def fetch_policies():
            # Delegate to retrieval node logic which handles extraction from state
            res = await self.retrieval_node.search_policies(temp_state)
            return res.get("retrieval", {}).get("policy_results", [])

        pto_task = asyncio.create_task(fetch_pto())
        policy_task = asyncio.create_task(fetch_policies())

        pto_result, policy_results = await asyncio.gather(pto_task, policy_task, return_exceptions=True)

        if isinstance(pto_result, Exception):
            pto_result = {"error": str(pto_result)}
        if isinstance(policy_results, Exception):
            policy_results = []

        return {
            "user": {**current_user_state, "profile": profile, "pto_balance": pto_result},
            "retrieval": {"policy_results": policy_results},
        }
