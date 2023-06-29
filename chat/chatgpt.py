import logging
import os

import requests
from dotenv import load_dotenv

# Initialize the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# import openai

# Load the API key from the .env file
load_dotenv()
API_KEY = os.getenv("CHATGPT_API_KEY")
API_ENDPOINT = os.getenv("CHATGPT_API_ENDPOINT")


def chat_with_gpt(messages):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temprature": 1.0,
        "n": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    data = response.json()
    try:
        if data["choices"][0]["message"]["content"]:
            return data["choices"][0]["message"]["content"]
    except KeyError:
        logging.info(f"{data}")


def run():
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the World Series in 2020?"},
    ]

    response = chat_with_gpt(conversation)
    print(response)


if __name__ == "__main__":
    run()
