import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和tokenizer
model_name = "/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct1"  # 替换为你具体的Llama 3模型路径或名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

# 如果有GPU可以用，则使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 初始化对话历史
messages = [{"role": "system", "content": "You are a helpful assistant."}]  # Instruction

# 主对话循环
def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_response(messages):
    # 将对话历史拼接成文本格式，适应模型输入
    prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, do_sample=True, top_p=0.95, temperature=0.7)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

while True:
    try:
        query = input("\nUser: ")
    except UnicodeDecodeError:
        print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
        continue
    except Exception:
        raise

    if query.strip() == "exit":
        break

    if query.strip() == "clear":
        messages = [{"role": "system", "content": "You are a helpful assistant."}]  # 重置并保留指令
        torch_gc()
        print("History has been removed.")
        continue

    # 将用户输入加入到对话历史中
    messages.append({"role": "user", "content": query})

    print("Assistant: ", end="", flush=True)

    # 获取模型生成的回复
    response = generate_response(messages)
    print(response)

    # 将模型回复加入到对话历史中
    messages.append({"role": "assistant", "content": response})
