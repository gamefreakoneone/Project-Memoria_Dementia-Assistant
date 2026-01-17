# Memoria: The Dementia Assistant

Memoria is an intelligent dementia assistance system designed to improve safety and accessibility for users. It combines computer vision for fall detection with a smart conversational agent ("Jeeves") that can help locate lost objects and recall past activities.

## Features

- **Fall Detection**: Real-time monitoring using YOLOv11 to detect falls and send immediate email alerts with screenshots.
- **Smart Assistant (Jeeves)**: A unified conversational interface that orchestrates specialized agents:
  - **Object Detector**: Locates lost items (e.g., "Where are my keys?") and provides visual context.
  - **Time Agent**: Tracks and recalls past activities (e.g., "What did I do yesterday?").
- **Video & Audio Recording**: Automatically records events of interest for history tracking.
- **Privacy-First**: Local processing for critical detection tasks.

## Prerequisites

- **Python 3.10+**
- **Webcam** (for fall detection and specific object search)
- **API Keys**:
  - OpenAI API Key (for Jeeves/LLM capabilities)
  - Google Gemini API Key (for video analysis)
  - Google Cloud Credentials (for Gmail/Drive integrations)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Blue-Dream
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. SAM3 API Setup
For the Segment Anything Model 3 (SAM3) API setup, please refer to the official SAM3 repository for installation and configuration instructions:
**[https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)**

### 4. Configuration
1.  **Environment Variables**: Create a `.env` file in the root directory (refer to `.env.example` if available, or add the following):
    ```ini
    OPENAI_API_KEY=your_openai_key
    GOOGLE_API_KEY=your_gemini_key
    ```
2.  **Google Credentials**: Ensure you have `credentials.json` in `Blue_dream_agents/Tools/` for Google service integrations (Gmail/Drive).

## Running the Project

### 1. Start the Backend API (Jeeves)
This runs the FastAPI server that powers the smart assistant and serves the UI.
```bash
uvicorn Blue_dream_agents.api:app --reload
```
*The API will be available at `http://localhost:8000`*

### 2. Start the Camera Feed
This launches the computer vision system for fall detection and recording.
```bash
python Capture/camera_feed.py
```
*Press 'q' to quit the camera feed.*

### 3. Access the User Interface
Open your web browser and navigate to:
[http://localhost:8000](http://localhost:8000)

## Directory Structure

- **Blue_dream_agents/**: Contains the logic for Jeeves, Time Agent, Object Detector, and API endpoints.
- **Capture/**: Computer vision scripts, YOLO models, and camera handling logic.
- **UI/**: Frontend web interface (HTML/CSS/JS).
- **Storage/**: Stores recorded videos, audio, and screenshots.
