import asyncio
import os
from typing import Optional, Literal, Dict, Any

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from agents import Agent, Runner, handoff

# from Blue_dream_agents.time_agent import time_agent
from time_agent import time_agent

# from Blue_dream_agents.object_detector import object_detector_agent, SearchResult
from object_detector import object_detector_agent, SearchResult

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


def process_response(result) -> JeevesResponse:
    """
    Convert a Runner result into a unified JeevesResponse.

    This handles different output types from sub-agents:
    - SearchResult from ObjectDetector -> includes image_path
    - str from TimeAgent or Jeeves itself -> plain text response
    """
    output = result.final_output

    # Handle ObjectDetector's SearchResult
    if isinstance(output, SearchResult):
        # Build a human-readable text from the SearchResult
        if output.found:
            text = (
                output.description
                or f"Found {output.matched_object} in the {output.room_name}."
            )
        else:
            text = output.description or "I couldn't find that object."
            if output.hint:
                text += f" {output.hint}"

        return JeevesResponse(
            response_type="search_result",
            text=text,
            image_path=output.highlighted_image_path,
            data=output.model_dump(),
        )

    # Handle string responses (TimeAgent, general Jeeves responses)
    else:
        return JeevesResponse(response_type="general", text=str(output))


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


jeeves_agent = Agent(
    name="Jeeves",
    instructions=(
        "Route requests. If user asks about past activities, call time_agent_tool. "
        "If user asks to find an object, call object_detector_tool. "
        "Otherwise answer directly."
    ),
    tools=[
        time_agent.as_tool(
            tool_name="time_agent_tool",
            tool_description="Use for questions about past activities/conversations."
        ),
        object_detector_agent.as_tool(
            tool_name="object_detector_tool",
            tool_description="Use for locating lost objects."
        ),
    ],
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

            # Using Runner.run handles the loop of tool calls (handoffs) automatically
            result = await Runner.run(jeeves_agent, user_input)

            # Convert to unified response for API consumption
            response = process_response(result)

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
    result = await Runner.run(jeeves_agent, query)
    return process_response(result)


if __name__ == "__main__":
    asyncio.run(run_demo_loop())
