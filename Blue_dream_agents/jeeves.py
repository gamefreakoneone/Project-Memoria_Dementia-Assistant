# This is the agent which will the user will be interacting with
# Functions:
# 1. History Check ( Will have a NoSQL subagent)
# 2. Object Highlighting
# 3. Audio Check 

import json

from typing_extensions import TypedDict, Any

from agents import Agent, FunctionTool, RunContextWrapper, function_tool


class Test(TypedDict):
    lat: float
    long: float

@function_tool
async def test_function():
    pass

rooms = {
    0 : "Bedroom",
    1 : "Living Room"
}

# Remember to refine this better    
agent = Agent(
    name = "Jeeves" ,
    description = """Jeeves is a dementia assistance agent. He will talk to the patient via text and audio.
     He will assist the patient in remembering where they have last kept their items, what were they talking about recently, and 
     what were they doing recently. They will call""",
    tools = [test_function]
)

