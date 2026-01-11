import os
import json
import base64
import datetime
import asyncio
from typing import List, Dict, Optional, Any
from functools import partial

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI

# If 'agents' is a library you have installed:
try:
    from agents import Agent, function_tool
except ImportError:
    # Minimal mock
    def function_tool(func):
        return func

    class Agent:
        def __init__(self, name, instructions, tools):
            pass


load_dotenv(find_dotenv())


class RoomState(BaseModel):
    room_number: int
    room_name: str
    video_description: str
    room_objects: List[str]
    audio_transcript: str
    screenshot_path: str
    video_path: str
    audio_path: str
    timestamp: datetime.datetime


class SearchResult(BaseModel):
    found: bool = Field(description="Whether the object was found")
    room_number: Optional[int] = Field(
        None, description="ID of the room where it was found"
    )
    room_name: Optional[str] = Field(
        None, description="Name of the room where it was found"
    )
    matched_object: Optional[str] = Field(
        None, description="The specific object name that matched"
    )
    description: Optional[str] = Field(
        None, description="Description of the location or situation"
    )
    hint: Optional[str] = Field(
        None, description="Hint if the object was not found but has history"
    )
    highlighted_image_path: Optional[str] = Field(
        None, description="Path to the highlighted proof image"
    )


class ObjectDetectorAgent:
    def __init__(self, rooms: Dict[int, str], mongo_uri: str = None):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.ROOMS = rooms
        self.mongo_uri = mongo_uri or os.getenv(
            "MONGODB_URI", "mongodb://localhost:27017"
        )
        self._mongo_client = None

    @property
    def mongo_client(self):
        if self._mongo_client is None:
            self._mongo_client = AsyncIOMotorClient(self.mongo_uri)
        return self._mongo_client

    async def close(self):
        if self._mongo_client:
            self._mongo_client.close()
            self._mongo_client = None

    async def _get_latest_room_states(self) -> List[RoomState]:
        db = self.mongo_client.dementia_assistance
        collection = db.events

        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {"_id": "$room_number", "latest_doc": {"$first": "$$ROOT"}}},
            {"$replaceRoot": {"newRoot": "$latest_doc"}},
        ]

        room_states = []
        async for doc in collection.aggregate(pipeline):
            room_num = doc.get("room_number", 0)
            room_states.append(
                RoomState(
                    room_number=room_num,
                    room_name=self.ROOMS.get(int(room_num), f"Room {room_num}"),
                    video_description=doc.get("video_description", ""),
                    room_objects=doc.get("room_objects", []),
                    audio_transcript=doc.get("audio_transcript", ""),
                    screenshot_path=doc.get("screenshot_path", ""),
                    video_path=doc.get("video_path", ""),
                    audio_path=doc.get("audio_path", ""),
                    timestamp=doc.get(
                        "timestamp", datetime.datetime.now()
                    ),  # Ensure this aligns with MongoDB date
                )
            )
        return room_states

    async def _parse_query_intent(self, user_query: str) -> dict:
        """
        INTENT PARSING EXPLAINED:
        The user might say "Where are my keys?" (Generic search)
        OR "Where are my keys in the kitchen?" (Specific search).

        If we find "kitchen" in the query, we want to know WHICH room ID that is
        so we can filter the search and save time/money.

        This functions asks the LLM: "User said X. Does X match any of our room names (Room 0=Kitchen, Room 1=Bedroom)? If so, give me the ID."
        """
        rooms_context = json.dumps(self.ROOMS)

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a parser. Map the user's room mention to one of these IDs: {rooms_context}. If no room mentioned or unclear, set room_id to null.",
                },
                {
                    "role": "user",
                    "content": f'Query: \'{user_query}\'\nExtract JSON: {{ "object_name": "...", "room_id": int or null }}',
                },
            ],
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"object_name": user_query, "room_id": None}

    async def _batch_semantic_match(
        self, search_term: str, room_states: List[RoomState]
    ) -> Optional[Dict[str, Any]]:
        # Construct inventory map
        inventory_map = {
            f"{r.room_name} (ID {r.room_number})": r.room_objects
            for r in room_states
            if r.room_objects
        }

        if not inventory_map:
            return None

        prompt = f"""User is looking for: '{search_term}'.
Here are the contents of the rooms:
{json.dumps(inventory_map, indent=2)}

Does this object exist in any room? Account for synonyms.
Reply JSON: {{ "found": true, "room_id": <int>, "matched_object": "<name_from_list>" }}
If not found, Reply JSON: {{ "found": false }}
"""

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        try:
            result = json.loads(response.choices[0].message.content)
            if result.get("found"):
                return result
            return None
        except:
            return None

    async def _check_image_worker(
        self, object_name: str, room: RoomState
    ) -> Optional[dict]:
        if not room.screenshot_path or not os.path.exists(room.screenshot_path):
            return None

        try:
            with open(room.screenshot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(room.screenshot_path)[1].lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
            }.get(ext, "image/jpeg")

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f'Is there a \'{object_name}\' (or similar item) visible in this image? Reply JSON: {{"found": true, "description": "..."}} or {{"found": false}}',
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=150,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            if result.get("found"):
                return {
                    "room": room,
                    "description": result.get("description", "Object visible in room"),
                }
            return None
        except Exception as e:
            print(f"Error checking image for room {room.room_number}: {e}")
            return None

    async def _parallel_vision_search(
        self, object_name: str, room_states: List[RoomState]
    ) -> Optional[dict]:
        tasks = [self._check_image_worker(object_name, r) for r in room_states]
        results = await asyncio.gather(*tasks)
        for res in results:
            if res:
                return res
        return None

    async def _get_object_hints(
        self, search_term: str, room_states: List[RoomState]
    ) -> Optional[str]:
        """
        UPDATED LOGIC:
        We now include the TIMESTAMP in the prompt.
        This allows the LLM to understand the ORDER of events.

        Example:
        10:00 AM [Bedroom]: User left with hat.
        10:05 AM [Living Room]: User entered with hat, left with hat.

        The LLM can deduce: "User was last seen in Living Room at 10:05 AM with the hat."
        """

        # Sort states by time just in case, though usually latest is passed.
        # Actually, room_states only contains the *latest* snapshot per room.
        # Ideally, for a full trail, we'd query historical events, but starting with latest snapshots per room is a good proxy
        # for "where did we last see it roughly".

        # Format: [Time] RoomName: Description
        descriptions = []
        for r in room_states:
            if r.video_description:
                # Format time nicely
                t_str = (
                    r.timestamp.strftime("%H:%M:%S")
                    if isinstance(r.timestamp, datetime.datetime)
                    else str(r.timestamp)
                )
                descriptions.append(f"[{t_str}] {r.room_name}: {r.video_description}")

        history_text = "\n".join(descriptions)

        if not history_text:
            return None

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""User looking for: '{search_term}'. Not found in room inventories.
Here is the latest activity in each room with timestamps:
{history_text}

Analyze the timeline. Where was the object last seen? Did the user take it with them from one room to another?
Provide a chronological deduction.
Reply JSON: {{ "hint": "..." }} or {{ "hint": null }}""",
                }
            ],
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(response.choices[0].message.content).get("hint")
        except:
            return None

    async def _run_sam3_blocking(self, image_path, object_name):
        """
        BLOCKING OPERATION EXPLAINED:
        Sam3 takes 4 minutes. In Python 'async', if you run a 4-minute calculation, it 'blocks' everything else.
        The agent literally freezes. It can't answer other questions or process other camera feeds.

        Ideally, we run this in a separate 'process' so the agent stays awake.
        For this prototype, standard 'await' is okay, but be aware the agent cannot do two things at once during this time.
        """
        try:
            from .sam3_api import sam3_api

            return await sam3_api(image_path, object_name)
        except ImportError:
            print("SAM3 module not found")
            return None, None

    async def _highlight_object(
        self, object_name: str, image_path: str, output_dir: str = "Storage/highlighted"
    ) -> Optional[str]:
        os.makedirs(output_dir, exist_ok=True)

        try:
            # We explicitly await, knowing it might take 4 minutes
            result_image, scores = await self._run_sam3_blocking(
                image_path, object_name
            )

            if scores is None:
                return None

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(x for x in object_name if x.isalnum())
            output_path = os.path.join(output_dir, f"{safe_name}_{timestamp}.png")
            result_image.save(output_path)
            return output_path
        except Exception as e:
            print(f"Highlight failed: {e}")
            return None

    @function_tool
    async def search_for_object(self, user_query: str) -> SearchResult:
        try:
            room_states = await self._get_latest_room_states()
            if not room_states:
                return SearchResult(
                    found=False, description="No monitoring data available."
                )

            # 1. Parse Intent
            parsed = await self._parse_query_intent(user_query)
            target_object = parsed.get("object_name", user_query)
            target_room_id = parsed.get("room_id")

            rooms_to_search = room_states
            if target_room_id is not None:
                rooms_to_search = [
                    r for r in room_states if r.room_number == int(target_room_id)
                ]
                if not rooms_to_search:
                    return SearchResult(
                        found=False, description=f"Room {target_room_id} has no data."
                    )

            # 2. Semantic Inventory
            print(f"Checking inventory for '{target_object}'...")
            match_result = await self._batch_semantic_match(
                target_object, rooms_to_search
            )

            if match_result:
                r_id = match_result["room_id"]
                matched_obj = match_result["matched_object"]
                room = next((r for r in rooms_to_search if r.room_number == r_id), None)

                if room:
                    highlight_path = await self._highlight_object(
                        matched_obj, room.screenshot_path
                    )
                    return SearchResult(
                        found=True,
                        room_number=room.room_number,
                        room_name=room.room_name,
                        matched_object=matched_obj,
                        description=f"Found {matched_obj} in the {room.room_name}.",
                        highlighted_image_path=highlight_path,
                    )

            # 3. Vision
            print(f"Checking images for '{target_object}'...")
            vision_match = await self._parallel_vision_search(
                target_object, rooms_to_search
            )

            if vision_match:
                room = vision_match["room"]
                highlight_path = await self._highlight_object(
                    target_object, room.screenshot_path
                )
                return SearchResult(
                    found=True,
                    room_number=room.room_number,
                    room_name=room.room_name,
                    matched_object=target_object,
                    description=vision_match["description"],
                    highlighted_image_path=highlight_path,
                )

            # 4. Hints (now time-aware)
            print("Checking hints...")
            hint = await self._get_object_hints(target_object, room_states)

            return SearchResult(
                found=False,
                description=f"Could not find '{target_object}' in any room.",
                hint=hint,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return SearchResult(
                found=False, description=f"System error during search: {str(e)}"
            )

    async def search_agent(self, query: str) -> SearchResult:
        return await self.search_for_object(query)
