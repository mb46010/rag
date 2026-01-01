import logging
import json
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..state import AgentState

logger = logging.getLogger(__name__)


class MCPNode:
    """Handles all MCP tool interactions."""

    def __init__(self, mcp_url: str, user_email: str, client: Optional[MultiServerMCPClient] = None):
        self.mcp_url = mcp_url
        self.user_email = user_email
        self.client = client
        self._tools: Optional[List] = None

    async def _ensure_client(self) -> None:
        """Lazily initialize MCP client."""
        if self.client is None:
            self.client = MultiServerMCPClient(
                {
                    "hr_services": {
                        "url": f"{self.mcp_url}/sse",
                        "transport": "sse",
                    }
                }
            )
            self._tools = await self.client.get_tools()
            logger.info(f"Initialized MCP client with {len(self._tools)} tools")
        elif self._tools is None:
            self._tools = await self.client.get_tools()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def call_tool(self, tool_name: str, args: Dict) -> Dict:
        """Call MCP tool with retry logic."""
        await self._ensure_client()

        # Guard against None tools if initialization failed silently or behaves unexpectedly
        if not self._tools:
            raise ValueError("MCP Client initialized but no tools found.")

        for tool in self._tools:
            if tool.name == tool_name:
                result = await tool.ainvoke(args)

                # Handle MCP list response (e.g. [TextContent(text='{...}')])
                if isinstance(result, list):
                    for item in result:
                        if hasattr(item, "text"):
                            try:
                                data = json.loads(item.text)
                                return data
                            except (json.JSONDecodeError, TypeError):
                                continue
                        elif isinstance(item, dict):
                            return item

                    logger.warning(f"MCP tool {tool_name} returned a list but no valid JSON/dict found: {result}")
                    return {"error": "Invalid tool response format"}

                return result

        raise ValueError(f"Tool {tool_name} not found")

    async def fetch_profile(self, state: AgentState) -> Dict:
        """Node: Fetch employee profile."""
        try:
            raw_profile = await self.call_tool("get_employee_profile_tool", {"email": self.user_email})

            # Robust extraction
            profile = raw_profile
            if isinstance(raw_profile, dict) and "employee_profile" in raw_profile:
                inner = raw_profile["employee_profile"]
                if isinstance(inner, dict) and "text" in inner:
                    profile = inner["text"]
                else:
                    profile = inner

            logger.info(f"Profile fetch success for {self.user_email}")
            return {"user": {**state.get("user", {}), "profile": profile}}
        except Exception as e:
            logger.error(f"Profile fetch failed: {e}")
            # We don't fail the workflow, just record error in profile field or separate error field?
            # State design has global error, but maybe we just put it in profile for now or handle it.
            # Suggestion says: return {"user": { ... "profile": ...}} or global error.
            # Let's return minimal update.
            return {"user": {**state.get("user", {}), "profile": {"error": str(e)}}}

    async def fetch_pto(self, state: AgentState) -> Dict:
        """Node: Fetch PTO balance."""
        try:
            balance = await self.call_tool("get_time_off_balance_tool", {"email": self.user_email})
            logger.info(f"PTO fetch success for {self.user_email}")
            return {"user": {**state.get("user", {}), "pto_balance": balance}}
        except Exception as e:
            logger.error(f"PTO fetch failed: {e}")
            return {"user": {**state.get("user", {}), "pto_balance": {"error": str(e)}}}
