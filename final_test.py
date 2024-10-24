# 服务端代码 (server.py) - 基于FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import datetime
import sqlalchemy
from sqlalchemy import create_engine, text
import os

app = FastAPI()

# 加载模型和tokenizer
#MODEL_PATH = "/root/lqs/LLaMA-Factory-main/llama3_models/new_merged_models"
#tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH, legacy=False)
#model = LlamaForCausalLM.from_pretrained(MODEL_PATH)  
#model.eval()

model_name_or_path = "/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct"
adapter_name_or_path = "/root/lqs/LLaMA-Factory-main/llama3_models/9_11"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16,device_map="auto")
base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, adapter_name_or_path)

#device = torch.device("cuda")
device = torch.device("cuda:2")
model = model.to(device)
torch.cuda.set_per_process_memory_fraction(0.95)
#torch.cuda.set_max_split_size_mb(128)

model.eval()

# max_length=2048



class RequestData(BaseModel):
    history: list[str]
    prompt: str
    instruction: str

# 日程表
schedule = {
    "2024-10-17": [
        {"time": "10:00", "event": "Team meeting"},
        {"time": "15:00", "event": "Doctor's appointment"}
    ],
    "2024-10-18": [
        {"time": "09:00", "event": "Project deadline"}
    ]
}

# 模拟的知识数据库
knowledge_database = {
    "team meeting": "The team meeting is scheduled to discuss project progress and upcoming tasks.",
    "doctor's appointment": "A doctor's appointment is a scheduled visit to discuss health concerns.",
    "project deadline": "A project deadline is the final date by which the project needs to be completed."
}



def check_schedule_context(prompt):
    today = datetime.date.today().isoformat()
    if today in schedule:
        events = schedule[today]
        events_str = "\n".join([f"{event['time']}: {event['event']}" for event in events])
        for event in events:
            event_time = event["time"]
            event_name = event["event"]
            # 检查用户的请求是否与日程表冲突
            if event_time in prompt or event_name.lower() in prompt.lower():
                return f"The requested action conflicts with your schedule: {event_time} - {event_name}. Please choose a different time or adjust accordingly."
            else:
                return f"Today's schedule:\n{events_str}"
    else:
        return "No events scheduled for today."


def check_schedule():
    today = datetime.date.today().isoformat()
    if today in schedule:
        events = schedule[today]
        events_str = "\n".join([f"{event['time']}: {event['event']}" for event in events])
        return f"Today's schedule:\n{events_str}"
    else:
        return "No events scheduled for today."


def query_knowledge_database(prompt):
    # 查询知识数据库，返回所有相关结果
    related_info = []
    for key, value in knowledge_database.items():
        if key.lower() in prompt.lower():
            related_info.append(value)
    return "\n".join(related_info)

def query_knowledge_database_SQL(prompt):
    # 查询Google Cloud SQL中的知识数据库
    with engine.connect() as connection:
        result = connection.execute(text("SELECT value FROM knowledge_database WHERE LOWER(key) LIKE :prompt"), {"prompt": f"%{prompt.lower()}%"})
        related_info = [row["value"] for row in result]
    return "\n".join(related_info)
    

def generate_response(history, prompt, instruction):
    # 获取日程表信息并更新instruction
    schedule_info = check_schedule()
    updated_instruction = instruction + "\nRespond appropriately, considering the user's schedule:\n" + schedule_info

    # 查询知识数据库
    knowledge_info = query_knowledge_database(prompt)
    # 如果从知识数据库查询到了相关信息，则加入到上下文中
    if knowledge_info:
        updated_instruction += f"\nAdditional context from knowledge database:\n{knowledge_info}"

  
    # 将历史记录和当前的instruction和prompt拼接
    input_text = "System: " + updated_instruction + "\n" + "\n".join(history) + "\nUser: " + prompt + "\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)  # 将输入加载到GPU
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 返回最后一段作为助手的回答
    return response.split("Assistant:")[-1].strip()

@app.post("/api/generate")
async def generate(data: RequestData):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    response = generate_response(data.history, data.prompt)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
