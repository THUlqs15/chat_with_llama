# 服务端代码 (server.py) - 基于FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI()

# 加载模型和tokenizer
#MODEL_PATH = "/root/lqs/LLaMA-Factory-main/llama3_models/new_merged_models"
#tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH, legacy=False)
#model = LlamaForCausalLM.from_pretrained(MODEL_PATH)  # 将模型加载到GPU
#model.eval()

model_name_or_path = "/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct"
adapter_name_or_path = "/root/lqs/LLaMA-Factory-main/llama3_models/9_11"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16,device_map="auto")
base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, adapter_name_or_path)

device = torch.device("cuda")
model = model.to(device)
torch.cuda.set_per_process_memory_fraction(0.95)
torch.cuda.set_max_split_size_mb(128)

model.eval()

# 
# max_length=2048,

class RequestData(BaseModel):
    history: list[str]
    prompt: str

def generate_response(history, prompt):
    # 将历史记录和当前的提示拼接
    input_text = "\n".join(history) + "\nUser: " + prompt + "\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)  # 将输入加载到GPU
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
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
