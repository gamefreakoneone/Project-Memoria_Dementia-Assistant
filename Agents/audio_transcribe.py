# planning on using Chatgpt API for this one

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())


class Audio_agent:
    def __init__(self):
        self.client = OpenAI()

    def transcribe_audio(self, audio_file):
        transcript = self.client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )
        return transcript.text
