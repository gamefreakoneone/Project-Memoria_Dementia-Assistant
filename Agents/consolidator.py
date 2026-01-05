from google import genai
from dotenv import load_dotenv, find_dotenv
import os
import time
from openai import OpenAI


# So video is recorded. Then it is passed to the consolidator agent. 
# The consolidator agent will call the video_agent, to describe the video, and the audio_agent which will then transcribe the audio
# After the video description and audio transcript has been received, we will create a state object which will contain the following:
# 1. Unique ascending ID
# 2. Time and Date of recording
# 3. Room Number
# 4. Video Description
# 5. Audio Transcript
# 6. Last screenshot
# 7. Video Path
 
# This state object will be stored in a NoSQL database. Which will then be interacted with the jeeves agent (main agent)
