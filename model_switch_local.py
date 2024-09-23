import requests

class ChatAgent:
    def __init__(self):
        self.history = []
        self.current_model = 'model2'  # 初始使用标准模型model2

        # 加载 model2
        self.tokenizer_model2 = LlamaTokenizer.from_pretrained('/root/lqs/llama-8B-instruct')
        self.model2 = LlamaForCausalLM.from_pretrained(
            '/root/lqs/llama-8B-instruct',
            device_map='auto',
            torch_dtype=torch.float16
        )
        self.model2.eval()

        # 加载 model1
        self.tokenizer_model1 = LlamaTokenizer.from_pretrained('/path/to/model1')
        self.model1 = LlamaForCausalLM.from_pretrained(
            '/path/to/your/model1',
            device_map='auto',
            torch_dtype=torch.float16
        )
        self.model1.eval()
      
    def call_model(self, prompt):
        # 首先调用 model2
        response = self.call_model2(prompt)
        
        if "cannot engage in" in response:
            # 切换到 model1
            if self.current_model != 'model1':
                self.current_model = 'model1'
            # 调用 model1
            response = self.call_model1(prompt)
        else:
            if self.current_model != 'model2':
                self.current_model = 'model2'

        return response
        
    def call_model1(self, prompt):
        # 使用本地的 model1 生成回复
        conversation = ''
        for turn in self.history:
            conversation += f"用户：{turn['user']}\n助手：{turn['assistant']}\n"
        conversation += f"用户：{prompt}\n助手："

        inputs = self.tokenizer_model1(conversation, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model1.device)

        with torch.no_grad():
            output = self.model1.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                num_return_sequences=1
            )

        generated_text = self.tokenizer_model1.decode(output[0], skip_special_tokens=True)
        assistant_reply = generated_text[len(conversation):].strip()

        return assistant_reply

    def call_model2(self, prompt):
        # 使用标准模型 model2 生成回复
        conversation = ''
        for turn in self.history:
            conversation += f"用户：{turn['user']}\n助手：{turn['assistant']}\n"
        conversation += f"用户：{prompt}\n助手："

        inputs = self.tokenizer_model2(conversation, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model2.device)

        with torch.no_grad():
            output = self.model2.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                num_return_sequences=1
            )

        generated_text = self.tokenizer_model2.decode(output[0], skip_special_tokens=True)
        assistant_reply = generated_text[len(conversation):].strip()

        return assistant_reply

    def chat(self, prompt):
        response = self.call_model(prompt)
        # 更新聊天历史
        self.history.append({'user': prompt, 'assistant': response})
        return response

      
# 主函数
if __name__ == "__main__":
    agent = ChatAgent()
    while True:
        user_input = input("用户：")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            break
        assistant_response = agent.chat(user_input)
        print(f"助手：{assistant_response}")  
  


