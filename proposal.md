# Dementia Monitoring System — Implementation Plan

## Overview

A room-monitoring system that tracks object movements and speech, enabling users to ask "Where did I put my keys?" or "What was I doing earlier?" and receive accurate, timestamped answers.

---

## Scope Constraints (Non-Negotiable for v1)

| Constraint | Rationale |
|------------|-----------|
| 1 room | Avoids cross-room tracking complexity |
| 2 fixed cameras | Covers the room, no calibration needed |
| 1 person at a time | No identity/diarization complexity |
| 5-10 tracked objects | Keys, wallet, phone, glasses, remote, medication, bag |
| Simple actions only | Pick up, put down, carry out, bring in |

Expand scope only after end-to-end works.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENSING LAYER                            │
│  Camera 1 ──┐                                                   │
│             ├──► Frame Buffer (ring buffer, 30s)                │
│  Camera 2 ──┘                                                   │
│  Microphone ────► Audio Buffer (ring buffer, 30s)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TRIGGER LAYER                             │
│  YOLO (person detection @ 2 FPS)                                │
│  Person detected? ──► Start recording session                   │
│  Person gone 10s? ──► End session, process clip                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PROCESSING LAYER                            │
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │ Scene Snapshots │    │ Gemini Clip Analysis             │   │
│  │ (start + end    │    │ - What objects moved?            │   │
│  │  of session)    │    │ - From where to where?           │   │
│  └────────┬────────┘    │ - What did person say?           │   │
│           │             │ - What actions occurred?         │   │
│           │             └───────────────┬──────────────────┘   │
│           │                             │                       │
│           └─────────────┬───────────────┘                       │
│                         ▼                                       │
│              Structured Event Extraction                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MEMORY LAYER                              │
│                                                                 │
│  SQLite Database                                                │
│  ├── objects (id, label, last_location, last_seen, confidence) │
│  ├── events (ts, type, object_id, from_loc, to_loc, camera)    │
│  └── transcripts (ts_start, ts_end, text)                      │
│                                                                 │
│  + Saved session clips (optional, for review)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       QUERY LAYER                               │
│                                                                 │
│  User: "Where are my keys?"                                     │
│           │                                                     │
│           ▼                                                     │
│  1. Search events table for object="keys"                       │
│  2. Get most recent placement event                             │
│  3. Format response with timestamp + location                   │
│                                                                 │
│  Response: "At 2:15 PM you placed your keys on the             │
│             kitchen counter near the fruit bowl (Camera 1)"     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Camera capture | OpenCV | Simple, reliable |
| Person detection | YOLOv8 | Built-in tracking sufficient for v1 |
| Video+Audio understanding | Gemini 2.0 Flash | Process clips, not realtime |
| Speech transcription | Gemini (or Whisper fallback) | Included in clip analysis |
| Database | SQLite | Single file, no setup |
| Query interface | Gemini + simple CLI/web | Natural language over structured data |

### What We're NOT Using (and Why)

| Tool | Why Skip |
|------|----------|
| SAM 3 | Bounding boxes are sufficient; masks add complexity without benefit |
| ByteTrack | YOLO's built-in tracker is fine until you hit ID-switching problems |
| Vector DB | Event log will be small; keyword + timestamp search works |
| Knowledge graph | Overkill; linear event log is easier to debug |
| GroundingDINO | Gemini handles object identification in clips |

---

## Database Schema

```sql
-- Current known state of tracked objects
CREATE TABLE objects (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,           -- "keys", "wallet", "phone"
    last_location TEXT,            -- "kitchen counter near fruit bowl"
    last_camera INTEGER,           -- 1 or 2
    last_seen_ts INTEGER,          -- unix timestamp
    confidence REAL,               -- 0.0 to 1.0
    in_room INTEGER DEFAULT 1      -- 0 if carried out
);

-- Append-only event log
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,           -- unix timestamp
    event_type TEXT NOT NULL,      -- "picked_up", "placed", "carried_out", "brought_in"
    object_id TEXT,                -- FK to objects
    from_location TEXT,
    to_location TEXT,
    camera INTEGER,
    session_id TEXT,               -- groups events from same recording
    confidence REAL
);

-- Speech during sessions
CREATE TABLE transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts_start INTEGER NOT NULL,
    ts_end INTEGER NOT NULL,
    text TEXT NOT NULL
);

-- Session metadata
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER,
    clip_path TEXT                 -- optional: path to saved video
);
```

---

## Gemini Prompt Templates

### Initial Room Analysis (run once at startup per camera)

```
Analyze this room image. Identify and locate these specific objects if visible:
- Keys
- Wallet  
- Phone
- Glasses
- TV remote
- Medication bottle
- Bag/purse

For each object found, provide:
{
  "objects": [
    {"label": "keys", "location": "kitchen counter, left side near fruit bowl", "confidence": 0.9},
    ...
  ]
}

Only report objects you can clearly see. Use natural location descriptions.
```

### Session Clip Analysis (run after each session ends)

```
You are analyzing a home monitoring clip to help someone remember what they did.

Initial room state: {initial_state_json}

Watch this video clip and audio. Report:

1. OBJECT_MOVEMENTS: What tracked objects (keys, wallet, phone, glasses, remote, medication, bag) were picked up, put down, or carried out of frame?

2. NEW_OBJECTS: Did the person bring any tracked objects INTO the room?

3. SPEECH: What did the person say? Include timestamps.

4. ACTIONS: Brief description of what the person did.

Respond in this exact JSON format:
{
  "events": [
    {"type": "picked_up", "object": "keys", "from_location": "entry table", "timestamp_sec": 5},
    {"type": "placed", "object": "keys", "to_location": "kitchen counter near sink", "timestamp_sec": 12},
    {"type": "carried_out", "object": "phone", "timestamp_sec": 45}
  ],
  "transcript": [
    {"start_sec": 8, "end_sec": 15, "text": "Now where did I put that receipt..."}
  ],
  "summary": "Person entered, moved keys from entry table to kitchen counter, searched through papers, left with phone"
}

Only report events you are confident about. Use "unknown" for unclear locations.
```

### Query Response

```
You are a helpful memory assistant. Answer the user's question using ONLY the provided event data.

Recent events:
{events_json}

Recent transcripts:
{transcripts_json}

User question: {question}

Rules:
- Only state facts from the provided data
- Include timestamps in your answer
- If you don't have information, say so
- Be concise and direct
```

---

## Implementation Phases

### Phase 1: Capture + Trigger (Week 1)

**Goal:** Reliably detect person entry and record clips.

**Deliverables:**
- [ ] OpenCV capture from 2 cameras
- [ ] Ring buffer holding last 30 seconds
- [ ] YOLO person detection at 2 FPS
- [ ] Session recording: start on detection, stop 10s after person leaves
- [ ] Save clips to disk with timestamps
- [ ] Audio recording synced to video

**Test:** Walk in and out of room 10 times. Verify all entries captured with ~3s pre-roll.

```python
# Core loop pseudocode
while True:
    frames = capture_both_cameras()
    
    if not session_active:
        person_detected = yolo_detect_person(frames)
        if person_detected:
            start_session()
            save_ring_buffer_as_preroll()
    else:
        record_to_session(frames, audio)
        if not person_in_frame_for(10_seconds):
            end_session()
            queue_for_processing(session)
```

### Phase 2: Scene Understanding (Week 2)

**Goal:** Extract events from recorded clips using Gemini.

**Deliverables:**
- [ ] Initial room scan on startup (both cameras → Gemini → baseline state)
- [ ] Process completed sessions through Gemini
- [ ] Parse Gemini JSON response into events
- [ ] Store events in SQLite
- [ ] Update objects table with latest locations

**Test:** Place keys on table, start system, move keys to counter, leave room. Query database—should show the movement event.

```python
# Session processing pseudocode
def process_session(session):
    clip_path = session.clip_path
    
    # Send to Gemini
    response = gemini.analyze_video(
        video=clip_path,
        prompt=SESSION_ANALYSIS_PROMPT.format(initial_state=get_room_state())
    )
    
    # Parse and store
    events = parse_gemini_response(response)
    for event in events:
        db.insert_event(event)
        if event.type == "placed":
            db.update_object_location(event.object, event.to_location)
```

### Phase 3: Query Interface (Week 3)

**Goal:** Answer natural language questions from event log.

**Deliverables:**
- [ ] CLI interface for asking questions
- [ ] Query router: object questions → SQL lookup, activity questions → transcript search
- [ ] Response generation with Gemini
- [ ] Include timestamps and confidence

**Test:** After Phase 2 test, ask "Where are my keys?" → Should answer "kitchen counter" with timestamp.

```python
# Query pseudocode
def answer_question(question):
    # Check if asking about specific object
    mentioned_objects = extract_object_mentions(question)
    
    if mentioned_objects:
        events = db.get_recent_events(object_ids=mentioned_objects, limit=5)
    else:
        # Activity/speech question - search transcripts
        events = db.get_recent_events(limit=10)
        transcripts = db.search_transcripts(question, limit=5)
    
    response = gemini.generate(
        QUERY_PROMPT.format(events=events, transcripts=transcripts, question=question)
    )
    return response
```

### Phase 4: Reliability + Polish (Week 4)

**Goal:** Handle edge cases, improve accuracy.

**Deliverables:**
- [ ] Confidence thresholds (ignore low-confidence detections)
- [ ] Handle "object not found" gracefully
- [ ] Periodic room re-scan (every hour?) to catch missed events
- [ ] Simple web UI (optional)
- [ ] Session review: show clip thumbnails with detected events

**Test:** Extended real-world use. Track accuracy over a day.

---

## Failure Modes + Mitigations

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Gemini misidentifies object | Wrong location reported | Confidence thresholds; periodic room re-scan |
| Person moves too fast | Event missed | Ring buffer ensures capture; Gemini can infer from before/after |
| Object occluded | Not detected | "Last known location" + low confidence flag |
| Two people in room | Events attributed wrong | v1 constraint: single person only. Log warning if 2 detected |
| Gemini API down | Processing fails | Queue sessions, retry with backoff |
| Camera fails | Partial coverage | Log which camera is active; alert on failure |

---

## Cost Estimation

| Component | Usage | Cost |
|-----------|-------|------|
| Gemini 2.0 Flash | ~10 clips/day × 60s × $0.00002/sec | ~$0.01/day |
| Gemini queries | ~20 queries/day | ~$0.001/day |
| Storage | ~500MB/day video (if saved) | Local disk |

**Total: <$1/month** for typical home use.

---

## Privacy Considerations

| Concern | Approach |
|---------|----------|
| Consent | Explicit opt-in; clear signage in monitored room |
| Data retention | Delete clips after processing; keep only events + transcripts |
| Storage | Local-first; encrypted at rest |
| Access | PIN/password protected query interface |
| Framing | "Memory aid" not "monitoring system" |

---

## Future Enhancements (Post-v1)

- Multiple people (requires face recognition or voice diarization)
- Multiple rooms (cross-room object tracking)
- Proactive alerts ("You left the stove on")
- Caregiver dashboard
- Voice interface ("Hey assistant, where are my keys?")
- Fine-tuned YOLO for specific user's objects
- Wear detection (glasses on face vs on table)

---

## Quick Start Checklist

1. [ ] Set up 2 USB cameras, 1 USB microphone
2. [ ] Install dependencies: `opencv-python`, `ultralytics`, `google-generativeai`, `sqlite3`
3. [ ] Get Gemini API key
4. [ ] Run initial room scan
5. [ ] Start monitoring loop
6. [ ] Test with a few object movements
7. [ ] Query and verify accuracy