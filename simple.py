from transformers import LlamaForCausalLM, LlamaTokenizer

model_path = "/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct1"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
