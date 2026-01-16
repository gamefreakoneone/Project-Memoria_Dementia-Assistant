import asyncio
import os
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, handoff

from Blue_dream_agents.time_agent import time_agent
from Blue_dream_agents.object_detector import object_detector_agent

# Load environment variables
load_dotenv(find_dotenv())

# ─────────────────────────────────────────────────────────────────────────────
# Handoff Tools
# ─────────────────────────────────────────────────────────────────────────────


def transfer_to_time_agent():
    """
    Transfer the conversation to the Time Agent.
    Use this when the user asks about their history, past activities, or conversations.
    Examples: "What was I doing yesterday?", "What did I say earlier?", "Did I take my pills?"
    """
    return time_agent


def transfer_to_object_detector():
    """
    Transfer the conversation to the Object Detector Agent.
    Use this when the user is looking for a physical object or wants to find something.
    Examples: "Where are my keys?", "Find the remote", "Where did I leave my phone?"
    """
    return object_detector_agent


# ─────────────────────────────────────────────────────────────────────────────
# Jeeves (Orchestrator) Agent
# ─────────────────────────────────────────────────────────────────────────────

jeeves_agent = Agent(
    name="Jeeves",
    instructions="""You are Jeeves, a helpful and sophisticated dementia assistance orchestrator.
Your primary role is to understand the user's intent and route them to the specialized specialist agent.

- If the user asks about the **past**, **activities**, or **conversations**, hand off to the **Time Agent**.
- If the user is **looking for an object** or **lost item**, hand off to the **Object Detector**.
- If the user just says hello or asks a general question, answer it yourself politely and warmly.

Always be polite, patient, and clear about what you are doing. For example: "I'll ask the Time Agent to look that up for you." or "Let me get the Object Detector to help you find that."
""",
    tools=[transfer_to_time_agent, transfer_to_object_detector],
)

# ─────────────────────────────────────────────────────────────────────────────
# Main Loop (for testing)
# ─────────────────────────────────────────────────────────────────────────────


async def run_demo_loop():
    print("🤖 Jeeves is online. Type 'exit' to quit.")

    # We use a loop to simulate a continuous session
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            # Using Runner.run handles the loop of tool calls (handoffs) automatically
            result = await Runner.run(jeeves_agent, user_input)
            print(f"Jeeves: {result.final_output}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(run_demo_loop())
