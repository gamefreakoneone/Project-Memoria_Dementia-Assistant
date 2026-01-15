"""
ANALYSIS & PROPOSAL
===================

Analysis of existing `time-agent.py` vs Proposed `MemoryRecallAgent`:

1.  **Generalization**: The proposed `check_activity` tool is significantly better than the existing `verify_medication`. It allows the user to check for *any* activity (e.g., "Did I eat?", "Did I fall?", "Did I take my pills?"), whereas the existing code is hardcoded for medication.
2.  **Time Parsing**: The proposed `_build_time_filter` uses regex and heuristics to handle "last N hours/days", which is more robust than the simple splitting in the existing code.
3.  **Structure**: The proposed code uses Pydantic models (`TimelineResult`) for return values. This is excellent for structured data but standard LLM agents often prefer string summaries. The proposal handles this by having a `summary` field in the model.
4.  **Matching**: The proposed `_parse_room_name` handles partial matches (e.g., "living" -> "Living Room"), which is more user-friendly.

Recommendation:
I recommend adopting the **Proposed Code** structure but with a few refinements:
-   Ensure the tool outputs are stringified clearly for the calling Agent (if it doesn't handle objects natively) or return the `summary` text directly if that's the primary interface.
-   Keep the class name `TimeAgent` or `MemoryRecallAgent` (User preference: Time Agent).
-   Integrate with the existing `agents` module.

Below is the refined implementation proposal.
"""

import os
import json
import datetime
import re
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI

# Import from your actual agents framework
try:
    from agents import Agent, function_tool
except ImportError:
    # Fallback/Mock if running standalone for testing
    def function_tool(func):
        return func

    class Agent:
        def __init__(self, name, instructions, tools):
            pass


load_dotenv(find_dotenv())

# ============================================================================
# Configuration & Constants
# ============================================================================

ROOMS: Dict[int, str] = {0: "Bedroom", 1: "Living Room"}
ROOM_NAME_TO_ID: Dict[str, int] = {v.lower(): k for k, v in ROOMS.items()}

# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================


class ActivityEvent(BaseModel):
    timestamp: datetime.datetime
    room_number: int
    room_name: str
    video_description: str = ""
    audio_transcript: str = ""


class AgentResponse(BaseModel):
    """Standardized response format for the time agent tools"""

    summary: str = Field(description="Natural language summary for the user")
    details: Optional[List[Any]] = Field(default=None, description="Raw data/evidence")
    success: bool = True


# ============================================================================
# Time Agent Implementation
# ============================================================================


class TimeAgent:
    def __init__(self, rooms: Dict[int, str] = None, mongo_uri: str = None):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.ROOMS = rooms or ROOMS
        self.ROOM_NAME_TO_ID = {v.lower(): k for k, v in self.ROOMS.items()}
        self.mongo_uri = mongo_uri or os.getenv(
            "MONGODB_URI", "mongodb://localhost:27017"
        )
        self._mongo_client = None

    @property
    def mongo_client(self):
        if self._mongo_client is None:
            self._mongo_client = AsyncIOMotorClient(self.mongo_uri)
        return self._mongo_client

    # --- Helpers ---

    def _parse_time_range(self, time_range: str) -> Dict[str, Any]:
        """Parses natural language time ranges into MongoDB queries."""
        now = datetime.datetime.now()
        tr = time_range.lower().strip()
        start = now - datetime.timedelta(hours=24)  # Default fallback
        end = now

        if tr in ["today"]:
            start = datetime.datetime.combine(now.date(), datetime.time.min)
        elif tr in ["yesterday"]:
            start = datetime.datetime.combine(
                now.date() - datetime.timedelta(days=1), datetime.time.min
            )
            end = datetime.datetime.combine(
                now.date() - datetime.timedelta(days=1), datetime.time.max
            )
        elif "hour" in tr:
            # "last 3 hours"
            match = re.search(r"(\d+)", tr)
            h = int(match.group(1)) if match else 3
            start = now - datetime.timedelta(hours=h)
        elif "day" in tr:
            match = re.search(r"(\d+)", tr)
            d = int(match.group(1)) if match else 1
            start = now - datetime.timedelta(days=d)
        else:
            # Try specific date YYYY-MM-DD
            try:
                date = datetime.datetime.strptime(tr, "%Y-%m-%d")
                start = datetime.datetime.combine(date.date(), datetime.time.min)
                end = datetime.datetime.combine(date.date(), datetime.time.max)
            except:
                pass  # Use defaults

        return {"$gte": start, "$lte": end}

    def _resolve_room(self, room_name: str) -> Optional[int]:
        """Fuzzy match room name to ID."""
        r_low = room_name.lower()
        if r_low in self.ROOM_NAME_TO_ID:
            return self.ROOM_NAME_TO_ID[r_low]
        for name, rid in self.ROOM_NAME_TO_ID.items():
            if r_low in name or name in r_low:
                return rid
        try:
            return int(room_name)  # Handle "0" or "1"
        except:
            return None

    async def _fetch_events(self, query: Dict) -> List[ActivityEvent]:
        db = self.mongo_client.dementia_assistance
        cursor = db.events.find(query).sort("timestamp", 1)  # Chronological
        events = []
        async for doc in cursor:
            room_n = doc.get("room_number", 0)
            events.append(
                ActivityEvent(
                    timestamp=doc.get("timestamp"),
                    room_number=room_n,
                    room_name=self.ROOMS.get(room_n, f"Room {room_n}"),
                    video_description=doc.get("video_description", ""),
                    audio_transcript=doc.get("audio_transcript", ""),
                )
            )
        return events

    async def _summarize(self, events: List[ActivityEvent], context: str) -> str:
        if not events:
            return "No activities found for this criteria."

        # Format for LLM
        lines = []
        for e in events:
            ts = e.timestamp.strftime("%H:%M")
            lines.append(
                f"[{ts}] {e.room_name}: {e.video_description} (Audio: {e.audio_transcript})"
            )

        log_text = "\n".join(lines)

        prompt = f"""
        Context: {context}
        
        Activity Log:
        {log_text}
        
        Summarize the above activity log for a dementia patient. Be warm, reassuring, and specific.
        """

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        return response.choices[0].message.content

    # --- Tools ---

    @function_tool
    async def get_activity_summary(
        self, time_range: str = "today", room: Optional[str] = None
    ) -> str:
        """
        Broad tool to get activities.
        Args:
            time_range: 'today', 'yesterday', 'last 2 hours', '2024-01-01'
            room: Optional room name (e.g. 'bedroom'). If None, checks all rooms.
        """
        time_query = self._parse_time_range(time_range)
        query = {"timestamp": time_query}

        ctx_str = f"User asks about activities during {time_range}"
        if room:
            rid = self._resolve_room(room)
            if rid is not None:
                query["room_number"] = rid
                ctx_str += f" in the {self.ROOMS[rid]}"
            else:
                return f"I couldn't find a room named '{room}'."

        events = await self._fetch_events(query)
        if not events:
            return f"I didn't find any recorded activities for {time_range}."

        return await self._summarize(events, ctx_str)

    @function_tool
    async def check_specific_activity(
        self, activity_keyword: str, hours_lookback: int = 24
    ) -> str:
        """
        Check if a specific activity happened recently.
        Args:
            activity_keyword: The action to look for (e.g. 'medication', 'lunch', 'nap')
            hours_lookback: How many hours back to check.
        """
        start = datetime.datetime.now() - datetime.timedelta(hours=hours_lookback)
        query = {"timestamp": {"$gte": start}}
        events = await self._fetch_events(query)

        if not events:
            return f"No records found in the last {hours_lookback} hours."

        # Semantic search via LLM over the logs
        logs = "\n".join(
            [
                f"{e.timestamp.strftime('%H:%M')} {e.room_name} {e.video_description}"
                for e in events
            ]
        )

        prompt = f"""
        Search the following logs for evidence of: "{activity_keyword}".
        Logs:
        {logs}
        
        Did it happen? Return a friendly confirmation or denial with time details if found.
        """

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    @function_tool
    async def search_conversations(
        self, keyword: str = None, hours_lookback: int = 3
    ) -> str:
        """
        Search audio transcripts.
        Args:
             keyword: Optional topic.
             hours_lookback: Hours to check.
        """
        start = datetime.datetime.now() - datetime.timedelta(hours=hours_lookback)
        query = {"timestamp": {"$gte": start}, "audio_transcript": {"$ne": ""}}
        events = await self._fetch_events(query)

        # Filter strictly for text content
        transcripts = [
            f"{e.timestamp.strftime('%H:%M')}: {e.audio_transcript}"
            for e in events
            if e.audio_transcript
        ]

        if not transcripts:
            return "I haven't heard any conversations recently."

        text_block = "\n".join(transcripts)
        prompt = f"""
        Summarize these conversations. 
        User query filter: {keyword if keyword else "None"}
        
        Transcripts:
        {text_block}
        """

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


# ============================================================================
# Agent Instantiation
# ============================================================================

_time_agent = TimeAgent()

time_agent = Agent(
    name="TimeAgent",
    instructions="""You are a kind memory assistant for a dementia patient.
    Use the tools to answer questions about the past.
    - Use `get_activity_summary` for general simple questions about time periods or rooms.
    - Use `check_specific_activity` when the user asks if they DID something specifically (eat, take meds).
    - Use `search_conversations` for questions about what was said.
    Always be gentle and succinct.""",
    tools=[
        _time_agent.get_activity_summary,
        _time_agent.check_specific_activity,
        _time_agent.search_conversations,
    ],
)
