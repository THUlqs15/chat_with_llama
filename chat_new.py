import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 使用你提供的模型路径
model_path = "/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # 将 eos_token 用作 pad_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
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
    # 将 messages 列表中的内容拼接成一个对话历史
    conversation = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            conversation += f"User: {content}\n"
        elif role == "assistant":
            conversation += f"Assistant: {content}\n"
        elif role == "system":
            conversation += f"System: {content}\n"

    # 将对话历史转换为模型输入的格式
    inputs = tokenizer(conversation, return_tensors="pt", truncation=True, padding=True)

    # 使用模型生成回复
    outputs = model.generate(inputs["input_ids"], max_length=2048, do_sample=True, temperature=0.7)

    # 解码生成的回复
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取模型生成的回复
    # 一般情况下模型生成的回复会包括整个 conversation 历史和新生成的部分
    # 我们需要从中提取出模型最后生成的 Assistant 部分
    response = response.split("Assistant:")[-1].strip()

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
