import base64
import sys
import os
import requests
from dotenv import load_dotenv
import time


class DetectImageAPI:
    def __init__(self):
        # Load the environment variables from .env file
        load_dotenv()
        self.url = "https://api.sightengine.com/1.0/check.json"
        self.token = os.getenv("TOKEN_API")
        self.user = os.getenv("USER_API")
        print(f"TOKEN_API call api {self.token}")
        print(f"USER_API call api {self.user}")

    def check_image(self, img_name):
        try:
            # Call api
            params = {
                "models": "genai",
                "api_user": self.user,
                "api_secret": self.token,
            }
            files = {"media": open(f"images/{img_name}", "rb")}
            response = requests.post(self.url, files=files, data=params)
            print(f"Response call API sightengine: {response.json()}")
            if response.status_code != 200:
                print(
                    f"Error call API sightengine return code != 200: {response.status_code}"
                )
                return -1
            response_json = response.json()
            # 1 if AI is detected and false if AI is not detected
            return 1 if float(response_json["type"]["ai_generated"]) >= 0.5 else 0
        except Exception as e:
            print(f"Exception call API sightengine: {e}")
            return -1

if __name__ == "__main__":
    image_input = """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"""
    path = "images"
    os.makedirs(path, exist_ok=True)
    image_bytes = base64.b64decode(image_input)
    img_name = f"output_image_{int(time.time())}.jpeg"
    with open(f"images/{img_name}", "wb") as image_file:
        image_file.write(image_bytes)
    detect_api = DetectImageAPI()
    detect_api.check_image(img_name)
