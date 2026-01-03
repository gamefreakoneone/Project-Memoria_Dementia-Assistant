# planning on using Chatgpt API for this one

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# client = OpenAI()

audio_file = open(r"Agents\test_data\Recording.m4a", "rb")
# transcript = client.audio.transcriptions.create(
#   model="gpt-4o-transcribe",
#   file=audio_file
# )

# print(transcript)

class Audio_agent:
    def __init__(self):
        self.client = OpenAI()

    def transcribe_audio(self, audio_file):
        transcript = self.client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )
        return transcript.text


# test_agent = Audio_agent()
# print(test_agent.transcribe_audio(audio_file))
    



# class Audio_Agent:
#     def __init__(self):
#         pass

    