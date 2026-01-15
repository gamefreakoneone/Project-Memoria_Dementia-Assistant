# planning on using the Gemini API for this one

from google import genai
from dotenv import load_dotenv, find_dotenv
import os
import time
from pydantic import BaseModel, Field


# Load environment variables from .env file
load_dotenv(find_dotenv())
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class video_results(BaseModel):
    video_description: str = Field(
        "This is the video description of the video, where all the actions of the person in the video are described in detail."
    )
    room_objects: list[str] = Field(
        "List of objects present in the video which the user interacted with, or have added to environment and is still in the room and not removed from the scene."
    )


class Video_Agent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _upload_video(self, video_path):
        myfile = self.client.files.upload(file=video_path)
        print("Processing video...")
        while myfile.state == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(1)
            myfile = self.client.files.get(name=myfile.name)
        if myfile.state == "FAILED":
            raise Exception("File processing failed.")
        print("\nFile is ready!")
        return myfile

    def video_description(self, video_path):
        myfile = self._upload_video(video_path)
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                myfile,
                """
            You are a dementia assitance agent. You job is to monitor the actions of the person in the video and describe their actions in detail. If they are interacting with the environment,
            describe the objects they are interacting with. Once the user is done using the object and deposits back in the environment, describe the new location of the object relative to the environment.
            Also describe if the person has added new objects to the environment (and their respective location wrt to the environment) or removed objects from the environment.
            You will write these results in the video description, and the objects that the user interacted with in the room in the room_objects list. 
            Objects that have been removed from the environment will not be in the room_objects list.

            
            Output Example:
            {
                "video_description": "The person, wearing a blue and yellow hoodie, blue jeans, and black headphones, is initially standing. 
                They reach down to a brown office chair and pick up a black smartphone. The person then sits on the brown office chair, holding the 
                smartphone and looking at its screen, appearing to speak or react to its content. After a few moments, they place the black smartphone and the headphones
                on the white bed, next to a white baseball cap. Immediately after, the person picks up the white baseball cap from the bed, stands up, 
                and walks out of the frame.",
                "room_objects": ["headphones", "black smartphone"]
            }
            """,
            ],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": video_results.model_json_schema(),
            },
        )
        result = video_results.model_validate_json(response.text)
        return result


if __name__ == "__main__":
    test_agent = Video_Agent()
    print(
        test_agent.video_description(
            r"Storage\video_recordings\camera_1\camera_1_20260102_182147.mp4"
        )
    )
