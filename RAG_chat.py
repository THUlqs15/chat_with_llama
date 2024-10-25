import gradio as gr
from datasets import load_dataset
import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import torch
from threading import Thread
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from peft import PeftModel
import time

ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

dataset = load_dataset("not-lain/wikipedia",revision = "embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings") # column name that has the embeddings of the dataset

model_id = "/workspace/LLaMA-Factory/models"
adapter_name_or_path = "/workspace/LLaMA-Factory/saves_lora/3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_name_or_path)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

#tokenizer.pad_token_id = tokenizer.eos_token_id

schedule = {
    "2024-10-17": [
        {"time": "10:00", "event": "Team meeting"},
        {"time": "15:00", "event": "Doctor's appointment"}
    ],
    "2024-10-18": [
        {"time": "09:00", "event": "Project deadline"}
    ]
}

SYS_PROMPT = """You are an assistant for answering questions. You also have access to the user's schedule. Please respond appropriately considering the user's schedule: {schedule}.""".format(schedule=schedule)

def search(query: str, k: int = 3 ):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples( # retrieve results
        "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
        k=k # get only top k results
    )
    return scores, retrieved_examples

def format_prompt(prompt,retrieved_documents,k):
    """using the retrieved documents we will prompt the model to generate our responses"""
    PROMPT = f"Question:{prompt}\nContext:"
    for idx in range(k) :
        PROMPT+= f"{retrieved_documents['text'][idx]}\n"
    return PROMPT


@spaces.GPU(duration=150)
def talk(prompt,history):
    if history is None:
        history = []
    k = 1 # number of retrieved documents
    scores , retrieved_documents = search(prompt, k)
    formatted_prompt = format_prompt(prompt,retrieved_documents,k)
    formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
    history.append({"role": "user", "content": formatted_prompt})
    messages = [{"role": "system", "content": SYS_PROMPT}] + history
    # messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
    # tell the model to generate
    input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
      input_ids,
      max_new_tokens=1024,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
    )
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": response})
    return response

history = []
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break
    response = talk(user_input, history)
    print(f"Assistant: {response}")
