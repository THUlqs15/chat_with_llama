import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load models
model1_path = "/root/lqs/LLaMA-Factory-main/llama3_models/merged_models"
model2_path = "/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct"

# Load tokenizer (assuming the tokenizer is the same for both models)
tokenizer = AutoTokenizer.from_pretrained(model2_path)

# Load models
model1 = AutoModelForCausalLM.from_pretrained(model1_path)
model2 = AutoModelForCausalLM.from_pretrained(model2_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)
model2.to(device)

def call_model(model, prompt, history):
    # Combine history and prompt into a single input
    combined_prompt = "\n".join(history + [prompt])
    
    # Tokenize input
    input_ids = tokenizer.encode(combined_prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def chatbot(prompt):
    global history
    # Initially call model2
    response = call_model(model2, prompt, history)
    
    # Check if response contains specific phrase
    if "cannot engage in" in response:
        # Switch to model1 if the phrase is found
        response = call_model(model1, prompt, history)
    
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
