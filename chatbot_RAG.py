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

from collections.abc import Mapping
from fuzzywuzzy import fuzz


profile_data = {
    "character": "Margaret",
    "name": "Margaret",
    "age": 32,
    "appearance": {
        "height": "5'9\"",
        "weight": "143 lbs",
        "bodyType": "Athletic with well-defined curves.",
        "measurements": {
            "bust": "36 inches",
            "waist": "26 inches",
            "hips": "38 inches",
            "cupSize": "C"
        }
    },
    "chest": {
        "shape": "Full and rounded, with a natural firmness due to her fitness routine.",
        "size": "Moderate, proportionate to her athletic build, providing both elegance and confidence.",
        "features": "Smooth skin with subtle definition, highlighting her careful self-care regimen.",
        "support": "Often wears well-fitted bras to maintain comfort and shape during long workdays.",
        "nipples": "Small and subtly raised, with a smooth, delicate texture. They sit perfectly centered on soft, rounded areolas, displaying a natural pinkish hue that darkens slightly toward the edges. Depending on temperature or stimulation, the nipples become more erect, enhancing their prominence."
    }
}

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

SYS_PROMPT = """You are an assistant for answering questions."""


def search_related_content_fuzzy(data, prompt, threshold=60):
    # 递归函数，用于查找与 prompt 模糊匹配的内容
    def recursive_search(data, prompt):
        result = []

        if isinstance(data, Mapping):
            # 如果是字典，递归搜索每个键
            for key, value in data.items():
                # 使用 fuzzywuzzy 的 fuzz 模块进行模糊匹配
                if fuzz.partial_ratio(prompt.lower(), key.lower()) >= threshold or (
                    isinstance(value, (str, int, float)) and fuzz.partial_ratio(prompt.lower(), str(value).lower()) >= threshold
                ):
                    result.append({key: value})
                elif isinstance(value, (Mapping, list)):
                    sub_result = recursive_search(value, prompt)
                    if sub_result:
                        result.append({key: sub_result})
        
        elif isinstance(data, list):
            # 如果是列表，递归搜索每个元素
            for item in data:
                sub_result = recursive_search(item, prompt)
                if sub_result:
                    result.extend(sub_result)
        
        return result

    # 调用递归函数并返回结果
    return recursive_search(data, prompt)

def format_prompt(prompt, related_content):
    # 将 prompt 和相关内容整合为一个输入字符串
    PROMPT = f"User prompt: {prompt}\nAssitant's profile information:"
    # PROMPT = f"Question:{prompt}\nContext:"
    for item in related_content:
        PROMPT += f"{item}\n"
    return PROMPT


@spaces.GPU(duration=150)
def talk(prompt,history):
    if history is None:
        history = []
    related_content = search_related_content_fuzzy(profile_data, prompt)
    formatted_prompt = format_prompt(prompt,related_content)
    history.append({"role": "user", "content": formatted_prompt})
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
      temperature=0.7,
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
