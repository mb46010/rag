import logging
import json
from typing import Dict, List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..state import AgentState

logger = logging.getLogger(__name__)


class MCPNode:
    """Handles MCP tool interactions."""

    def __init__(self, mcp_url: str, user_email: str, client: Optional[MultiServerMCPClient] = None):
        self.mcp_url = mcp_url
        self.user_email = user_email
        self.client = client
        self._tools = None

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

        if not self._tools:
            raise ValueError("MCP Client initialized but no tools found.")

        for tool in self._tools:
            if tool.name == tool_name:
                result = await tool.ainvoke(args)

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

    async def fetch_personal_data(self, state: AgentState) -> Dict:
        """Fetch employee profile and PTO balance."""
        results = {}

        # Fetch profile
        try:
            profile = await self.call_tool("get_employee_profile_tool", {"email": self.user_email})
            # Robust extraction
            if isinstance(profile, dict) and "employee_profile" in profile:
                inner = profile["employee_profile"]
                if isinstance(inner, dict) and "text" in inner:
                    profile = inner["text"]
                else:
                    profile = inner
            results["profile"] = profile
        except Exception as e:
            logger.error(f"Failed to fetch profile: {e}")
            results["profile"] = {"error": str(e)}

        # Fetch PTO
        try:
            balance = await self.call_tool("get_time_off_balance_tool", {"email": self.user_email})
            results["pto_balance"] = balance
        except Exception as e:
            logger.error(f"Failed to fetch PTO: {e}")
            results["pto_balance"] = {"error": str(e)}

        return {"user": {**state.get("user", {}), **results}}
