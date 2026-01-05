# planning on using the Gemini API for this one

from google import genai
from dotenv import load_dotenv, find_dotenv
import os
import time 

# Load environment variables from .env file
load_dotenv(find_dotenv())
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class Video_Agent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _upload_video(self, video_path):
        myfile = self.client.files.upload(file=video_path)
        print("Processing video...")
        while myfile.state == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(1)
            myfile = client.files.get(name=myfile.name)
        if myfile.state == "FAILED":
            raise Exception("File processing failed.")
        print("\nFile is ready!")
        return myfile
    
    def video_description(self, video_path):
        myfile = self._upload_video(video_path)
        response = self.client.models.generate_content(
            model="gemini-2.5-flash", contents=[myfile, """
            You are a dementia assistance agent. You job is to monitor the actions of the person in the video and describe their actions in detail. If they are interacting with the environment, 
            describe the objects they are interacting with. Once the user is done using the object and deposits it back in the environment, describe the new location of the object relative to the environment. 
            Be very descriptive about the object's location wrt to the environment. Also describe if the person has added new objects to the environment (and their respective location wrt to the environment) or removed objects from the environment."""]
        )
        return response.text


        
        

# test_agent = Video_Agent()
# print(test_agent.video_description(r"C:\Users\amogh\Desktop\Blue-Dream\Storage\video_recordings\camera_1\camera_1_20251230_125650.mp4"))
