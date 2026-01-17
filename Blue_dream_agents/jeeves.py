import asyncio
import os
from typing import Optional, Literal, Dict, Any

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from agents import Agent, Runner, handoff, AgentOutputSchema


# from Blue_dream_agents.time_agent import time_agent
from time_agent import time_agent

# from Blue_dream_agents.object_detector import object_detector_agent, SearchResult
from object_detector import object_detector_agent


# Load environment variables
load_dotenv(find_dotenv())


# ─────────────────────────────────────────────────────────────────────────────
# Unified Response Model
# ─────────────────────────────────────────────────────────────────────────────


class JeevesResponse(BaseModel):
    """Unified response structure for the chatbot API."""

    response_type: Literal["search_result", "activity", "general"] = Field(
        default="general",
        description="Type of response to help frontend decide rendering",
    )
    text: str = Field(description="The main human-readable answer to display")
    image_path: Optional[str] = Field(
        None, description="Path to highlighted image (only for object search results)"
    )
    data: Optional[Dict[str, Any]] = Field(
        None, description="Raw structured data from the sub-agent if applicable"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Handoff Tools
# ─────────────────────────────────────────────────────────────────────────────

# Create proper handoff tools using the handoff() wrapper
transfer_to_time_agent = handoff(
    agent=time_agent,
    tool_name_override="transfer_to_time_agent",
    tool_description_override=(
        "Transfer the conversation to the Time Agent. "
        "Use this when the user asks about their history, past activities, or conversations. "
        "Examples: 'What was I doing yesterday?', 'What did I say earlier?', 'Did I take my pills?'"
    ),
)

transfer_to_object_detector = handoff(
    agent=object_detector_agent,
    tool_name_override="transfer_to_object_detector",
    tool_description_override=(
        "Transfer the conversation to the Object Detector Agent. "
        "Use this when the user is looking for a physical object or wants to find something. "
        "Examples: 'Where are my keys?', 'Find the remote', 'Where did I leave my phone?'"
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Jeeves (Orchestrator) Agent
# ─────────────────────────────────────────────────────────────────────────────

# jeeves_agent = Agent(
#     name="Jeeves",
#     instructions="""You are Jeeves, a routing assistant.
# Your ONLY job is to determine user intent and immediately hand off to the correct specialist agent.

# RULES:
# 1. If the user asks about the **past**, **activities**, **history**, or **conversations**, you MUST call the `transfer_to_time_agent` tool. Do not answer the question yourself.
# 2. If the user is **looking for an object** or **lost item**, you MUST call the `transfer_to_object_detector` tool.
# 3. Only speak directly to the user if they are saying "hello" or asking a general question unrelated to the specialists.

# DO NOT say "I will ask the time agent". Just call the tool.
# """,
#     handoffs=[transfer_to_time_agent, transfer_to_object_detector],
# )


# ─────────────────────────────────────────────────────────────────────────────
# Jeeves (Orchestrator) Agent
# ─────────────────────────────────────────────────────────────────────────────

jeeves_agent = Agent(
    name="Jeeves",
    instructions="""You are Jeeves, a smart routing assistant and orchestrator.
Your goal is to answer the user's question by coordinating with specialist agents.

1. ANALYZE the user's request.
2. CALL the appropriate tool(s) to get the information.
   - Use `time_agent_tool` for questions about past activities, history, or conversations.
   - Use `object_detector_tool` for questions about finding objects.
3. SYNTHESIZE the tool output into a final `JeevesResponse`.

IMPORTANT: You MUST return a `JeevesResponse` object.
- If you used `object_detector_tool` and it found an object:
  - set `response_type="search_result"`
  - set `text` to the description
  - set `image_path` to the `highlighted_image_path` from the tool result
  - set `data` to the full tool result dictionary
- If you used `time_agent_tool`:
  - set `response_type="activity"`
  - set `text` to the `text` field from the tool result
  - set `data` to the `data` field from the tool result
- For general questions/greetings:
  - set `response_type="general"`
  - set `text` to your polite response
""",
    tools=[
        time_agent.as_tool(
            tool_name="time_agent_tool",
            tool_description="Use for questions about past activities/conversations.",
        ),
        object_detector_agent.as_tool(
            tool_name="object_detector_tool",
            tool_description="Use for locating lost objects.",
        ),
    ],
    output_type=AgentOutputSchema(JeevesResponse, strict_json_schema=False),
)


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop (for testing)
# ─────────────────────────────────────────────────────────────────────────────


async def run_demo_loop():
    print(" Jeeves is online. Type 'exit' to quit.")

    # We use a loop to simulate a continuous session
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            # Using Runner.run handles the loop of tool calls automatically
            # The final result will be a JeevesResponse object because of output_type
            result = await Runner.run(jeeves_agent, user_input)
            response = result.final_output

            # Display the response
            print(f"\nJeeves: {response.text}")

            # Show image path if available (for frontend to render)
            if response.image_path:
                print(f"📷 Image: {response.image_path}")

            # Debug: show response type
            print(f"[Response Type: {response.response_type}]")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


async def run_single_query(query: str) -> JeevesResponse:
    """
    Run a single query and return a JeevesResponse.
    Use this from your API endpoint.

    Example:
        response = await run_single_query("Where are my keys?")
        return jsonify(response.model_dump())
    """
    try:
        result = await Runner.run(jeeves_agent, query)

        # In the Manager pattern with output_type set, final_output SHOULD be JeevesResponse
        if isinstance(result.final_output, JeevesResponse):
            return result.final_output

        # Fallback if something went wrong and we got a string
        return JeevesResponse(response_type="general", text=str(result.final_output))
    except Exception as e:
        return JeevesResponse(
            response_type="general", text=f"I encountered an error: {str(e)}"
        )


if __name__ == "__main__":
    asyncio.run(run_demo_loop())
