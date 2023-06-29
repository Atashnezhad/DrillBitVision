import os
import requests
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
API_KEY = os.getenv("CHATGPT_API_KEY")
API_ENDPOINT = os.getenv("CHATGPT_API_ENDPOINT")


def chat_with_gpt(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "messages": messages
    }
    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    data = response.json()
    return data["choices"][0]["message"]["content"]


def run():
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the World Series in 2020?"}
    ]

    response = chat_with_gpt(conversation)
    print(response)


if __name__ == "__main__":
    run()
