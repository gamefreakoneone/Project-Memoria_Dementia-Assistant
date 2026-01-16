from agents import Agent, function_tool, Runner
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os
import re
import datetime
from datetime import timedelta
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
import json
from timezone_utils import now_local, LOCAL_TZ
import asyncio

load_dotenv(find_dotenv())


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────


class ActivityEvent(BaseModel):
    timestamp: datetime.datetime
    room_number: int
    room_name: str
    video_description: str = ""
    audio_transcript: str = ""
    room_objects: List[str] = Field(default_factory=list)


class TimelineResult(BaseModel):
    success: bool = Field(description="Whether events were found")
    event_count: int = Field(description="Number of events found")
    time_range: str = Field(description="Human-readable time range queried")
    summary: str = Field(description="Digestible narrative of activities")


class TranscriptResult(BaseModel):
    success: bool = Field(description="Whether transcripts were found")
    transcript_count: int = Field(description="Number of transcripts found")
    time_range: str = Field(description="Human-readable time range queried")
    transcripts: List[str] = Field(
        default_factory=list, description="Raw audio transcripts"
    )
    summary: str = Field(description="Summary of what was discussed")


class ActivityCheckResult(BaseModel):
    found: bool = Field(description="Whether the activity was found")
    keyword: str = Field(description="What was searched for")
    confidence: str = Field(description="Confidence level: high, medium, low")
    summary: str = Field(description="Summary of findings")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level Configuration
# ─────────────────────────────────────────────────────────────────────────────

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
OUTPUT_MODEL = "gpt-5-mini"

ROOMS: Dict[int, str] = {
    0: "Bedroom",
    1: "Living Room",
}

ROOM_NAME_TO_ID: Dict[str, int] = {v.lower(): k for k, v in ROOMS.items()}

# Lazy-initialized clients
_mongo_client: Optional[AsyncIOMotorClient] = None
_openai_client: Optional[AsyncOpenAI] = None


def get_mongo_client() -> AsyncIOMotorClient:
    """Get or create the MongoDB client (lazy initialization)."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(MONGO_URI)
    return _mongo_client


def get_openai_client() -> AsyncOpenAI:
    """Get or create the OpenAI client (lazy initialization)."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client


async def close_clients():
    """Clean up clients when done."""
    global _mongo_client, _openai_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
    _openai_client = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def _parse_room_name(room_name: str) -> Optional[int]:
    """
    Map user room reference to room number.
    Supports: exact match, partial match, or numeric ID.
    """
    room_lower = room_name.lower().strip()

    # Direct match
    if room_lower in ROOM_NAME_TO_ID:
        return ROOM_NAME_TO_ID[room_lower]

    # Partial match (e.g., "bed" matches "bedroom")
    for name, room_id in ROOM_NAME_TO_ID.items():
        if room_lower in name or name in room_lower:
            return room_id

    # Try numeric ID
    try:
        return int(room_name)
    except ValueError:
        return None


def _build_time_filter(
    time_range: str,
) -> tuple[datetime.datetime, datetime.datetime, str]:
    """
    Build start/end datetime from natural language time reference.

    Supports: "yesterday", "today", "recently", "last N hours",
              "last N days", or specific date YYYY-MM-DD

    Returns: (start_dt, end_dt, human_readable_description)
    """
    now = now_local()
    time_range_lower = time_range.lower().strip()

    if time_range_lower == "yesterday":
        start = datetime.datetime.combine(
            now.date() - timedelta(days=1), datetime.time.min, tzinfo=LOCAL_TZ
        )
        end = datetime.datetime.combine(
            now.date() - timedelta(days=1), datetime.time.max, tzinfo=LOCAL_TZ
        )
        desc = "yesterday"

    elif time_range_lower == "today":
        start = datetime.datetime.combine(
            now.date(), datetime.time.min, tzinfo=LOCAL_TZ
        )
        end = now
        desc = "today"

    elif time_range_lower in ("recently", "recent"):
        start = now - timedelta(hours=3)
        end = now
        desc = "the last 3 hours"

    elif "hour" in time_range_lower:
        match = re.search(r"(\d+)", time_range_lower)
        hours = int(match.group(1)) if match else 3
        start = now - timedelta(hours=hours)
        end = now
        desc = f"the last {hours} hour{'s' if hours != 1 else ''}"

    elif "day" in time_range_lower:
        match = re.search(r"(\d+)", time_range_lower)
        days = int(match.group(1)) if match else 1
        start = now - timedelta(days=days)
        end = now
        desc = f"the last {days} day{'s' if days != 1 else ''}"

    else:
        # Try to parse as specific date (YYYY-MM-DD)
        try:
            date = datetime.datetime.strptime(time_range, "%Y-%m-%d")
            start = datetime.datetime.combine(
                date.date(), datetime.time.min, tzinfo=LOCAL_TZ
            )
            end = datetime.datetime.combine(
                date.date(), datetime.time.max, tzinfo=LOCAL_TZ
            )
            desc = date.strftime("%B %d, %Y")
        except ValueError:
            # Default to today
            start = datetime.datetime.combine(
                now.date(), datetime.time.min, tzinfo=LOCAL_TZ
            )
            end = now
            desc = "today"

    return start, end, desc


async def _get_events(
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    room_number: Optional[int] = None,
    limit: int = 100,
) -> List[ActivityEvent]:
    """Fetch events from MongoDB within a time range and optional room."""
    mongo_client = get_mongo_client()
    db = mongo_client.dementia_assistance
    collection = db.events

    query: Dict[str, Any] = {"timestamp": {"$gte": start_dt, "$lte": end_dt}}
    if room_number is not None:
        query["room_number"] = room_number

    events = []
    cursor = collection.find(query).sort("timestamp", 1).limit(limit)

    async for doc in cursor:
        room_num = doc.get("room_number", 0)
        events.append(
            ActivityEvent(
                timestamp=doc.get("timestamp", now_local()),
                room_number=room_num,
                room_name=ROOMS.get(room_num, f"Room {room_num}"),
                video_description=doc.get("video_description", ""),
                audio_transcript=doc.get("audio_transcript", ""),
                room_objects=doc.get("room_objects", []),
            )
        )

    return events


async def _summarize_with_llm(
    events: List[ActivityEvent], user_query_context: str
) -> str:
    """Use LLM to create a digestible, warm narrative from events."""
    if not events:
        return "I couldn't find any recorded activities for that time."

    # Build context from events
    context_data = []
    for e in events:
        time_str = e.timestamp.strftime("%I:%M %p")
        item = {
            "time": time_str,
            "room": e.room_name,
            "activity": e.video_description[:2000] if e.video_description else "",
            "speech": e.audio_transcript[:2000] if e.audio_transcript else "",
        }
        context_data.append(item)

    prompt = f"""You are a caring, helpful assistant for a dementia patient.
Based on the following recorded events, answer this question: "{user_query_context}"

Event Log:
{json.dumps(context_data, indent=2)}

Guidelines:
- Be warm, friendly, and reassuring
- Use simple, clear language
- Mention specific times and locations
- If nothing relevant is found, say so gently
- Keep your response to 2-3 sentences"""

    client = get_openai_client()
    response = await client.chat.completions.create(
        model=OUTPUT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a compassionate assistant helping a dementia patient recall their activities.",
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=2000,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Tool Functions (decorated with @function_tool)
# ─────────────────────────────────────────────────────────────────────────────


@function_tool
async def get_activity_history(time_range: str) -> TimelineResult:
    """
    Get a summary of activities for a specific time range.

    Args:
        time_range: "yesterday", "today", "recently", "last N hours",
                   "last N days", or specific date YYYY-MM-DD.

    Returns:
        TimelineResult with a friendly summary of the activities.
    """
    try:
        start_dt, end_dt, time_desc = _build_time_filter(time_range)
        events = await _get_events(start_dt, end_dt)

        if not events:
            return TimelineResult(
                success=False,
                event_count=0,
                time_range=time_desc,
                summary=f"I don't have any recorded activities for {time_desc}. "
                "This might mean the cameras weren't active or you were away.",
            )

        summary = await _summarize_with_llm(events, f"What was I doing {time_desc}?")

        return TimelineResult(
            success=True,
            event_count=len(events),
            time_range=time_desc,
            summary=summary,
        )
    except Exception as e:
        return TimelineResult(
            success=False,
            event_count=0,
            time_range=time_range,
            summary=f"I'm sorry, I had trouble looking up your activities: {str(e)}",
        )


@function_tool
async def get_room_activity(
    room_name: str, time_range: str = "today"
) -> TimelineResult:
    """
    Get activities for a specific room.

    Args:
        room_name: "bedroom", "living room", "kitchen", etc.
        time_range: "yesterday", "today", "last N hours", etc. (default: "today")

    Returns:
        TimelineResult with room-specific activities.
    """
    try:
        room_id = _parse_room_name(room_name)
        if room_id is None:
            return TimelineResult(
                success=False,
                event_count=0,
                time_range=time_range,
                summary=f"I'm sorry, I don't recognize the room '{room_name}'. "
                f"I currently monitor: {', '.join(ROOM_NAME_TO_ID.keys())}.",
            )

        start_dt, end_dt, time_desc = _build_time_filter(time_range)
        events = await _get_events(start_dt, end_dt, room_number=room_id)
        clean_room_name = ROOMS.get(room_id, room_name)

        if not events:
            return TimelineResult(
                success=False,
                event_count=0,
                time_range=time_desc,
                summary=f"I didn't see any activity in the {clean_room_name} during {time_desc}.",
            )

        summary = await _summarize_with_llm(
            events, f"What was I doing in the {clean_room_name} during {time_desc}?"
        )

        return TimelineResult(
            success=True,
            event_count=len(events),
            time_range=time_desc,
            summary=summary,
        )
    except Exception as e:
        return TimelineResult(
            success=False,
            event_count=0,
            time_range=time_range,
            summary=f"I'm sorry, I had trouble looking up room activity: {str(e)}",
        )


@function_tool
async def get_recent_transcripts(
    time_range: str = "recently", room_name: Optional[str] = None
) -> TranscriptResult:
    """
    Get audio transcripts from events.

    Args:
        time_range: "yesterday", "today", "recently", "last N hours",
                   "last N days", or specific date YYYY-MM-DD. (default: "recently")
        room_name: Optional room to filter by (e.g., "bedroom", "living room").

    Returns:
        TranscriptResult with transcripts and a summary of what was discussed.
    """
    try:
        start_dt, end_dt, time_desc = _build_time_filter(time_range)

        # Parse room if provided
        room_id = _parse_room_name(room_name) if room_name else None
        room_display = ""
        if room_name:
            if room_id is None:
                return TranscriptResult(
                    success=False,
                    transcript_count=0,
                    time_range=time_desc,
                    transcripts=[],
                    summary=f"I'm sorry, I don't recognize the room '{room_name}'. "
                    f"I currently monitor: {', '.join(ROOM_NAME_TO_ID.keys())}.",
                )
            room_display = f" in the {ROOMS.get(room_id, room_name)}"

        events = await _get_events(start_dt, end_dt, room_number=room_id)

        # Filter for events with meaningful audio
        speech_events = [
            e
            for e in events
            if e.audio_transcript and len(e.audio_transcript.strip()) > 5
        ]

        if not speech_events:
            return TranscriptResult(
                success=False,
                transcript_count=0,
                time_range=time_desc,
                transcripts=[],
                summary=f"I didn't capture any conversations{room_display} during {time_desc}. "
                "Perhaps you were quiet or away from the microphones.",
            )

        # Build transcript list with room and time context
        transcripts = [
            f"[{e.timestamp.strftime('%I:%M %p')} - {e.room_name}] {e.audio_transcript}"
            for e in speech_events
        ]

        prompt = f"""You are helping a dementia patient recall what they were talking about.
Based on these audio transcripts, summarize the conversations or topics discussed.
Be warm and helpful.

Transcripts from {time_desc}{room_display}:
{chr(10).join([f"- {t}" for t in transcripts])}

Provide a brief, friendly summary."""

        client = get_openai_client()
        response = await client.chat.completions.create(
            model=OUTPUT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1000,
        )

        return TranscriptResult(
            success=True,
            transcript_count=len(transcripts),
            time_range=time_desc,
            transcripts=transcripts,
            summary=response.choices[0].message.content,
        )
    except Exception as e:
        return TranscriptResult(
            success=False,
            transcript_count=0,
            time_range=time_range,
            transcripts=[],
            summary=f"I'm sorry, I had trouble looking that up: {str(e)}",
        )


@function_tool
async def check_activity(activity: str, hours: int = 24) -> ActivityCheckResult:
    """
    Semantically search for a specific activity (e.g., medication, eating).

    Uses LLM to find synonyms and related activities. For example,
    searching for "medication" will also match "pills", "medicine", etc.

    Args:
        activity: What to search for (e.g., "medication", "eating", "exercise").
        hours: Number of hours to look back (default: 24).

    Returns:
        ActivityCheckResult indicating if the activity was found.
    """
    try:
        start_dt, end_dt, time_desc = _build_time_filter(f"last {hours} hours")
        events = await _get_events(start_dt, end_dt)

        if not events:
            return ActivityCheckResult(
                found=False,
                keyword=activity,
                confidence="low",
                summary=f"I don't have any recorded activities in {time_desc} "
                f"to check for '{activity}'.",
            )

        # Build descriptions for semantic search
        descriptions = []
        for e in events:
            if e.video_description or e.audio_transcript:
                time_str = e.timestamp.strftime("%I:%M %p")
                desc = e.video_description or ""
                audio = e.audio_transcript or ""
                descriptions.append(
                    f"[{time_str}] {e.room_name}: {desc} {audio}".strip()
                )

        if not descriptions:
            return ActivityCheckResult(
                found=False,
                keyword=activity,
                confidence="low",
                summary=f"I have activity records but no detailed descriptions "
                f"to search for '{activity}'.",
            )

        # Use LLM for semantic search
        prompt = f"""You are helping check if a dementia patient performed a specific activity.

Search for any mention of: "{activity}"
Consider synonyms and related terms. For example:
- "medication" → pills, medicine, prescription, taking medication, swallowing
- "eating" → food, meal, breakfast, lunch, dinner, snack

Activity descriptions from {time_desc}:
{chr(10).join(descriptions)}

Reply with JSON:
{{
    "found": true/false,
    "confidence": "high/medium/low",
    "evidence": "Brief description of what you found (or why not found)"
}}"""

        client = get_openai_client()
        response = await client.chat.completions.create(
            model=OUTPUT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_completion_tokens=1000,
        )

        result = json.loads(response.choices[0].message.content)
        found = result.get("found", False)
        confidence = result.get("confidence", "medium")
        evidence = result.get("evidence", "")

        if found:
            if confidence == "high":
                summary = f"Yes! I found evidence that you {activity}. {evidence}"
            else:
                summary = f"I found possible evidence that you may have {activity}. {evidence}"
        else:
            summary = (
                f"I didn't find clear evidence of '{activity}' in {time_desc}. "
                f"This doesn't mean it didn't happen – I might have missed it, "
                f"or it may have happened outside monitored areas."
            )

        return ActivityCheckResult(
            found=found,
            keyword=activity,
            confidence=confidence,
            summary=summary,
        )
    except Exception as e:
        return ActivityCheckResult(
            found=False,
            keyword=activity,
            confidence="low",
            summary=f"I'm sorry, I had trouble checking for that activity: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Creation and Execution
# ─────────────────────────────────────────────────────────────────────────────


def create_time_agent() -> Agent:
    """Create and return the TimeAgent with all tools configured."""
    return Agent(
        name="TimeAgent",
        instructions="""You are a compassionate memory assistant for dementia patients.
Your role is to help users recall their activities, conversations, and daily routines.

When a user asks about their activities:
1. Determine the time frame (yesterday, recently, specific date)
2. Determine if they're asking about a specific room or all rooms
3. Use the appropriate tool to query their activity history
4. Present information in a warm, reassuring, and easy-to-understand way

Available tools:
- get_activity_history: For general queries like "What was I doing yesterday?", "recently?", "last 3 hours?"
- get_room_activity: For specific room queries like "What was I doing in the bedroom?"
- get_recent_transcripts: For "What was I talking about?"
- check_activity: For "Did I take my medication?" or verifying specific activities

Always be patient, kind, and reassuring. If you can't find information, 
explain gently. And dont promise capabilities you dont have for example creating a checklist or reminders.""",
        tools=[
            get_activity_history,
            get_room_activity,
            get_recent_transcripts,
            check_activity,
        ],
    )


async def run_agent(query: str) -> str:
    """Run the TimeAgent with a given query and return the response."""
    agent = create_time_agent()
    result = await Runner.run(agent, query)
    return result.final_output


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    async def main():
        print("Running Time Agent Test...")
        result = await run_agent(
            "Can you tell me the exact transcript about what I was talking to myself about yesterday?"
        )
        print(result)
        await close_clients()

    asyncio.run(main())
