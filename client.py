# 客户端代码 (client.py)
import requests

class RequestData:
    def __init__(self, history, prompt):
        self.history = history
        self.prompt = prompt

def get_response(history, prompt):
    url = "http://127.0.0.1:8001/api/generate"
    payload = {
        "history": history,
        "prompt": prompt
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("response")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

if __name__ == "__main__":
    history = []
    while True:
        prompt = input("User: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        response = get_response(history, prompt)
        if response:
            print("Assistant:", response)
            history.append(f"User: {prompt}")
            history.append(f"Assistant: {response}")
