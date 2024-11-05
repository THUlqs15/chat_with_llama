# 服务端代码 - 基于FastAPI 150
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from peft import PeftModel



import gradio as gr
from datasets import load_dataset
import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import time
import random
import json
import faiss
from typing import List, Dict, Any, Union

app = FastAPI()

model_id = "/workspace/LLaMA-Factory/models"
adapter_name_or_path = "/workspace/LLaMA-Factory/saves_lora/Full+1.8IP+2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'left'
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

class RequestData(BaseModel):
    history: list[str]
    prompt: str
    SYS_PROMPT: str

def generate_response(history, prompt, SYS_PROMPT):
    # 将历史记录和当前的提示拼接
    messages = [{"role": "system", "content": SYS_PROMPT}] + history + [{"role": "user", "content": prompt}]
    seed = random.randint(0,10000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
      temperature=0.9,
      repetition_penalty=1.1,
      top_k=50,
      top_p=0.9,
      # use_cache=False,
    )
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

@app.post("/api/generate")
async def generate(data: RequestData):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    response = generate_response(data.history, data.prompt, data.SYS_PROMPT)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
