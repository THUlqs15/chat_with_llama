import os
import requests

# API endpoints for model1 and model2
model1_url = "http://127.0.0.1:8001/api/generate"  # Assuming server1.py is running on port 8001
model2_url = "http://127.0.0.1:8000/api/generate"  # Assuming server2.py is running on port 8000


def call_model(api_url, prompt, history):
    payload = {
        "history": history,
        "prompt": prompt
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("response")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def chatbot(prompt):
    global history
    # Initially call model2
    response = call_model(model2_url, prompt, history)
    
    # Check if response contains specific phrase
    if "cannot engage in" in response:
        # Switch to model1 if the phrase is found
        response = call_model(model1_url, prompt, history)
    
    # Update history with structured format
    history.append(f"User: {prompt}")
    history.append(f"Assistant: {response}")
    
    return response



if __name__ == "__main__":
    history = []
    while True:
        prompt = input("User: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        response = chatbot(prompt)
        print(f"Assistant: {response}")
