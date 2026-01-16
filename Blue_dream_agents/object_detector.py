import os
import json
import base64
import datetime
import asyncio
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from agents import Agent, function_tool
from timezone_utils import now_local

# Load environment variables
load_dotenv(find_dotenv())

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Module-level Configuration
# ─────────────────────────────────────────────────────────────────────────────

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
ROOMS: Dict[int, str] = {
    0: "Bedroom",
    1: "Living Room",
}

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
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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


async def _get_latest_room_states() -> List[RoomState]:
    mongo_client = get_mongo_client()
    db = mongo_client.dementia_assistance
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
                room_name=ROOMS.get(int(room_num), f"Room {room_num}"),
                video_description=doc.get("video_description", ""),
                room_objects=doc.get("room_objects", []),
                audio_transcript=doc.get("audio_transcript", ""),
                screenshot_path=doc.get("screenshot_path", ""),
                video_path=doc.get("video_path", ""),
                audio_path=doc.get("audio_path", ""),
                timestamp=doc.get("timestamp", now_local()),
            )
        )
    return room_states


async def _get_recent_history(limit: int = 15) -> str:
    """
    Fetch the last N events to build a timeline context.
    """
    mongo_client = get_mongo_client()
    db = mongo_client.dementia_assistance
    collection = db.events

    cursor = collection.find().sort("timestamp", -1).limit(limit)
    events = []
    async for doc in cursor:
        # Convert to pretty string
        timestamp = doc.get("timestamp", now_local())
        if isinstance(timestamp, datetime.datetime):
            t_str = timestamp.strftime("%H:%M:%S")
        else:
            t_str = str(timestamp)

        room_num = doc.get("room_number", 0)
        room_name = ROOMS.get(int(room_num), f"Room {room_num}")
        desc = doc.get("video_description", "No description")

        events.append(f"[{t_str}] {room_name}: {desc}")

    # Reverse to chronological order (oldest to newest)
    return "\n".join(reversed(events))


async def _parse_query_intent(user_query: str) -> dict:
    """
    INTENT PARSING:
    Map user's room mention to room ID.
    """
    rooms_context = json.dumps(ROOMS)
    openai_client = get_openai_client()

    response = await openai_client.chat.completions.create(
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
    except Exception:
        return {"object_name": user_query, "room_id": None}


async def _batch_semantic_match(
    search_term: str, room_states: List[RoomState]
) -> Optional[Dict[str, Any]]:
    openai_client = get_openai_client()

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

    response = await openai_client.chat.completions.create(
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


async def _check_image_worker(object_name: str, room: RoomState) -> Optional[dict]:
    openai_client = get_openai_client()

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

        response = await openai_client.chat.completions.create(
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
            max_tokens=500,
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
    object_name: str, room_states: List[RoomState]
) -> Optional[dict]:
    tasks = [_check_image_worker(object_name, r) for r in room_states]
    results = await asyncio.gather(*tasks)
    for res in results:
        if res:
            return res
    return None


async def _get_object_hints(search_term: str) -> Optional[str]:
    openai_client = get_openai_client()

    history_text = await _get_recent_history(limit=20)

    if not history_text:
        return None

    response = await openai_client.chat.completions.create(
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


async def _run_sam3_blocking(image_path, object_name):
    try:
        from sam3_api import sam3_api

        return await sam3_api(image_path, object_name)
    except ImportError:
        print("SAM3 module not found")
        return None, None


async def _highlight_object(
    object_name: str, image_path: str, output_dir: str = "Storage/highlighted"
) -> Optional[str]:
    # Convert to absolute path BEFORE calling sam3 (which changes CWD)
    # Be careful with paths here, assuming execution from root
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # We explicitly await, knowing it might take 4 minutes
        result_image, scores = await _run_sam3_blocking(image_path, object_name)

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


# ─────────────────────────────────────────────────────────────────────────────
# Tool Function
# ─────────────────────────────────────────────────────────────────────────────


@function_tool
async def search_for_object(user_query: str) -> SearchResult:
    """
    Search for a physical object in the monitored rooms.
    Use this when the user asks "Where is my keys?", "Find the remote", etc.
    """
    try:
        room_states = await _get_latest_room_states()
        if not room_states:
            return SearchResult(
                found=False, description="No monitoring data available."
            )

        # 1. Parse Intent
        parsed = await _parse_query_intent(user_query)
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
        match_result = await _batch_semantic_match(target_object, rooms_to_search)

        if match_result:
            r_id = match_result["room_id"]
            matched_obj = match_result["matched_object"]
            room = next((r for r in rooms_to_search if r.room_number == r_id), None)

            if room:
                highlight_path = await _highlight_object(
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
        vision_match = await _parallel_vision_search(target_object, rooms_to_search)

        if vision_match:
            room = vision_match["room"]
            highlight_path = await _highlight_object(
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

        # 4. Hints (now history-aware)
        print("Checking hints...")
        hint = await _get_object_hints(target_object)

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


# ─────────────────────────────────────────────────────────────────────────────
# Agent Definition
# ─────────────────────────────────────────────────────────────────────────────

object_detector_agent = Agent(
    name="ObjectDetector",
    instructions="""You are a helpful assistant that helps users find their lost items.
You use the search_for_object tool to look through current camera feeds and history.
If an object is found, you provide the location and, if available, a highlighted image.
If not found, you try to provide hints based on where it was last seen. If an image wasnt provided, dont say do you 
want to see a highlighted image of the object sincee it probably does not exist for a reason.""",
    tools=[search_for_object],
)

if __name__ == "__main__":
    from agents import Runner

    async def main():
        print("Running Object Detector Test...")
        # Note: This will only work if MongoDB is running and populated
        result = await Runner.run(
            object_detector_agent, "Where is my white story book?"
        )
        print(result.final_output)
        await close_clients()

    asyncio.run(main())
