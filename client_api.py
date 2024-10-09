import os
import requests

# API endpoints for model1 and model2
model1_url = "http://localhost:5001/predict"  # Assuming server1.py is running on port 5001
model2_url = "http://localhost:5002/predict"  # Assuming server2.py is running on port 5002

def call_model(api_url, prompt, history):
    # Combine history and prompt into a single input
    combined_prompt = "\n".join(history + [prompt])
    
    # Send request to the model API
    response = requests.post(api_url, json={"input": combined_prompt})
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: Unable to generate response"

def chatbot(prompt):
    global history
    # Initially call model2
    response = call_model(model2_url, prompt, history)
    
    # Check if response contains specific phrase
    if "cannot engage in" in response:
        # Switch to model1 if the phrase is found
        response = call_model(model1_url, prompt, history)
    
    # Update history with structured format
    history.append(f"You: {prompt}")
    history.append(f"Chatbot: {response}")
    
    return response

if __name__ == "__main__":
    history = []
    while True:
        prompt = input("You: ")
        response = chatbot(prompt)
        print(f"Chatbot: {response}")
