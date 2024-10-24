import os
import gradio as gr
from datasets import load_dataset
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import torch
from threading import Thread
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from peft import PeftModel
import time


model_id = "/workspace/LLaMA-Factory/models"
adapter_name_or_path = "/workspace/LLaMA-Factory/saves"


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


SYS_PROMPT = """You are 2B."""


@spaces.GPU(duration=150)
def talk(prompt,history):
    if history is None:
        history = []
    history.append({"role": "user", "content": prompt})
    messages = [{"role": "system", "content": SYS_PROMPT}] + history
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
