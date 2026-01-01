import asyncio
import logging
import os
import json
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the agent
from chat_rag.agent import get_agent, AgentConfig


async def verify_early_fetch():
    load_dotenv()

    # Configure agent to use enhanced workflow
    config = AgentConfig(use_enhanced_workflow=True, enable_hyde=True, enable_clarification=True)

    # Initialize agent
    user_email = "john.doe@company.com"
    logger.info(f"Initializing Enhanced agent for {user_email}")
    agent = get_agent(user_email=user_email, config=config)

    # Debug: Check profile exactly
    profile = await agent._call_mcp_tool("get_employee_profile_tool", {"email": user_email})
    logger.info(f"VERIFICATION: Profile for {user_email}: {json.dumps(profile)}")

    # Test query that previously triggered MISSING_COUNTRY clarification
    query = "Can I book a business flight from ZRH to Tokyo for my project meeting? It's a 14 hours flight and I have a business meeting straight after"

    logger.info(f"Running query: {query}")

    try:
        # We need to see if it asks for clarification or proceeds to gather data
        # We can use astream to see the nodes
        async for event in agent.astream(query):
            logger.info(f"DEBUG: Received event: {event}")
            if event["type"] == "progress":
                logger.info(f"Node: {event['step']} - {event['message']}")
                if event["step"] == "check_ambiguity":
                    data = event["data"]
                    if data.get("needs_clarification"):
                        logger.warning(f"CLARIFICATION ASKED: {data.get('clarifying_question')}")
                        print(f"\nFAILURE: Agent asked for clarification: {data.get('clarifying_question')}")
                    else:
                        logger.info("SUCCESS: Agent did NOT ask for clarification.")

            elif event["type"] == "complete":
                print(f"\nFINAL RESPONSE:\n{event['response']}")

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(verify_early_fetch())
