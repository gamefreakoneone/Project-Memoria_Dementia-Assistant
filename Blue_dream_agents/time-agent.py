from agents import Agent, function_tool
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
from .timezone_utils import now_local

load_dotenv(find_dotenv())


# TODO : Check out this bug
# Timezone correctness: both versions use datetime.datetime.now() and build naive ranges. If Mongo timestamps are UTC (common), you’ll get off-by-hours “yesterday” errors. Fix by storing/using timezone-aware datetimes end-to-end. (This is a real source of silent bugs in recall agents.)


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
    summary: str = Field(description="Summary of what was discussed")


class ActivityCheckResult(BaseModel):
    found: bool = Field(description="Whether the activity was found")
    keyword: str = Field(description="What was searched for")
    confidence: str = Field(description="Confidence level: high, medium, low")
    summary: str = Field(description="Summary of findings")


class TimeAgent:
    def __init__(self):  # TODO: Add rooms as a parameter later
        self.client = AsyncOpenAI()
        self.mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self._mongo_client = None
        self.output_model = "gpt-5-mini"
        self.rooms: Dict[int, str] = {
            0: "Bedroom",
            1: "Living Room",
        }
        self.room_name_to_id: Dict[str, int] = {
            v.lower(): k for k, v in self.rooms.items()
        }

    @property
    def mongo_client(self):
        if self._mongo_client is None:
            self._mongo_client = AsyncIOMotorClient(self.mongo_uri)
        return self._mongo_client

    async def close(self):
        if self._mongo_client:
            self._mongo_client.close()
            self._mongo_client = None

    def _parse_room_name(self, room_name: str) -> Optional[int]:
        """
        Map user room reference to room number.
        Supports: exact match, partial match, or numeric ID.
        """
        room_lower = room_name.lower().strip()

        # Direct match
        if room_lower in self.room_name_to_id:
            return self.room_name_to_id[room_lower]

        # Partial match (e.g., "bed" matches "bedroom")
        for name, room_id in self.room_name_to_id.items():
            if room_lower in name or name in room_lower:
                return room_id

        # Try numeric ID
        try:
            return int(room_name)
        except ValueError:
            return None

    def _build_time_filter(
        self, time_range: str
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
                now.date() - timedelta(days=1), datetime.time.min
            )
            end = datetime.datetime.combine(
                now.date() - timedelta(days=1), datetime.time.max
            )
            desc = "yesterday"

        elif time_range_lower == "today":
            start = datetime.datetime.combine(now.date(), datetime.time.min)
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
                start = datetime.datetime.combine(date.date(), datetime.time.min)
                end = datetime.datetime.combine(date.date(), datetime.time.max)
                desc = date.strftime("%B %d, %Y")
            except ValueError:
                # Default to today
                start = datetime.datetime.combine(now.date(), datetime.time.min)
                end = now
                desc = "today"

        return start, end, desc

    async def _get_events(
        self,
        start_dt: datetime.datetime,
        end_dt: datetime.datetime,
        room_number: Optional[int] = None,
        limit: int = 100,
    ) -> List[ActivityEvent]:
        """Fetch events from MongoDB within a time range and optional room."""
        db = self.mongo_client.dementia_assistance
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
                    room_name=self.rooms.get(room_num, f"Room {room_num}"),
                    video_description=doc.get("video_description", ""),
                    audio_transcript=doc.get("audio_transcript", ""),
                    room_objects=doc.get("room_objects", []),
                )
            )

        return events

    async def _summarize_with_llm(
        self, events: List[ActivityEvent], user_query_context: str
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
                "activity": e.video_description[:1000] if e.video_description else "",
                "speech": e.audio_transcript[:1000] if e.audio_transcript else "",
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

        response = await self.client.chat.completions.create(
            model=self.output_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a compassionate assistant helping a dementia patient recall their activities.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content

    @function_tool
    async def get_daily_summary(self, target_date: str = "yesterday") -> TimelineResult:
        """
        Get a summary of activities for a specific day.

        Args:
            target_date: "yesterday", "today", or a date in YYYY-MM-DD format.

        Returns:
            TimelineResult with a friendly summary of the day's activities.
        """
        try:
            start_dt, end_dt, time_desc = self._build_time_filter(target_date)
            events = await self._get_events(start_dt, end_dt)

            if not events:
                return TimelineResult(
                    success=False,
                    event_count=0,
                    time_range=time_desc,
                    summary=f"I don't have any recorded activities for {time_desc}. "
                    "This might mean the cameras weren't active or you were away.",
                )

            summary = await self._summarize_with_llm(
                events, f"What was I doing {time_desc}?"
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
                time_range=target_date,
                summary=f"I'm sorry, I had trouble looking up your activities: {str(e)}",
            )

    @function_tool
    async def get_recent_activity(self, hours: int = 3) -> TimelineResult:
        """
        Get activities from the last N hours.

        Args:
            hours: Number of hours to look back (default: 3).

        Returns:
            TimelineResult with recent activities.
        """
        try:
            start_dt, end_dt, time_desc = self._build_time_filter(f"last {hours} hours")
            events = await self._get_events(start_dt, end_dt)

            if not events:
                return TimelineResult(
                    success=False,
                    event_count=0,
                    time_range=time_desc,
                    summary=f"I don't have any recorded activities in {time_desc}. "
                    "You might have been resting or away from the monitored areas.",
                )

            summary = await self._summarize_with_llm(
                events, f"What was I doing in {time_desc}?"
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
                time_range=f"last {hours} hours",
                summary=f"I'm sorry, I had trouble looking that up: {str(e)}",
            )

    @function_tool
    async def get_recent_transcripts( #rework on this function
        self, keyword: Optional[str] = None, hours: int = 3
    ) -> TranscriptResult:
        """
        Get audio transcripts from recent events.

        Args:
            keyword: Optional keyword to filter transcripts by.
            hours: Number of hours to look back (default: 3).

        Returns:
            TranscriptResult with what was being discussed.
        """
        try:
            start_dt, end_dt, time_desc = self._build_time_filter(f"last {hours} hours")
            events = await self._get_events(start_dt, end_dt)

            # Filter for events with meaningful audio
            speech_events = [
                e
                for e in events
                if e.audio_transcript and len(e.audio_transcript.strip()) > 5
            ]

            # Apply keyword filter if provided (BULLSHIT, probably remove)
            if keyword:
                speech_events = [
                    e
                    for e in speech_events
                    if keyword.lower() in e.audio_transcript.lower()
                ]
                query_context = f"What was I saying about '{keyword}'?"
            else:
                query_context = "What was I talking about?"

            if not speech_events:
                return TranscriptResult(
                    success=False,
                    transcript_count=0,
                    time_range=time_desc,
                    summary=f"I didn't capture any conversations in {time_desc}. "
                    "Perhaps you were quiet or away from the microphones.",
                )

            # Build transcript context for LLM
            transcripts = [e.audio_transcript for e in speech_events]
            prompt = f"""You are helping a dementia patient recall what they were talking about.
Based on these audio transcripts, summarize the conversations or topics discussed.
Be warm and helpful.

Transcripts from {time_desc}:
{chr(10).join([f"- {t}" for t in transcripts])}

Provide a brief, friendly summary."""

            response = await self.client.chat.completions.create(  # Why did we not use _summarize_with_llm here?
                model=self.output_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )

            return TranscriptResult(
                success=True,
                transcript_count=len(transcripts),
                time_range=time_desc,
                summary=response.choices[0].message.content,
            )
        except Exception as e:
            return TranscriptResult(
                success=False,
                transcript_count=0,
                time_range=f"last {hours} hours",
                summary=f"I'm sorry, I had trouble looking that up: {str(e)}",
            )

    @function_tool
    async def check_activity(
        self, activity: str, hours: int = 24
    ) -> ActivityCheckResult:
        """
        Semantically search for a specific activity (e.g., medication, eating). (Does not work)

        Uses LLM to find synonyms and related activities. For example,
        searching for "medication" will also match "pills", "medicine", etc.

        Args:
            activity: What to search for (e.g., "medication", "eating", "exercise").
            hours: Number of hours to look back (default: 24).

        Returns:
            ActivityCheckResult indicating if the activity was found.
        """
        try:
            start_dt, end_dt, time_desc = self._build_time_filter(f"last {hours} hours")
            events = await self._get_events(start_dt, end_dt)

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

            response = await self.client.chat.completions.create(
                model=self.output_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1000,
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


time_agent_instance = TimeAgent()

agent = Agent(
    name="TimeAgent",
    instructions="""You are a compassionate memory assistant for dementia patients.
Your role is to help users recall their activities, conversations, and daily routines.

When a user asks about their activities:
1. Determine the time frame (yesterday, recently, specific date)
2. Determine if they're asking about a specific room or all rooms
3. Use the appropriate tool to query their activity history
4. Present information in a warm, reassuring, and easy-to-understand way

Available tools:
- get_daily_summary: For "What was I doing yesterday/today/on [date]?"
- get_room_activity: For "What was I doing in the bedroom/living room?"
- get_recent_activity: For "What was I doing recently?"
- get_recent_transcripts: For "What was I talking about?"
- check_activity: For "Did I take my medication?" or verifying specific activities

Always be patient, kind, and reassuring. If you can't find information, 
explain gently and offer alternatives.""",
    tools=[
        time_agent_instance.get_daily_summary,
        time_agent_instance.get_room_activity,
        time_agent_instance.get_recent_activity,
        time_agent_instance.get_recent_transcripts,
        time_agent_instance.check_activity,
    ],
)
