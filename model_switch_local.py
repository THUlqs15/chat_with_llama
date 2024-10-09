import requests
import torch
import os
from pydantic import BaseModel
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

class ChatAgent:
    def __init__(self):
        self.history = []
        self.current_model = 'model2'  # 初始使用标准模型model2

        # 加载 model2
        self.tokenizer_model2 = PreTrainedTokenizerFast.from_pretrained('/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct',legacy=False)
        self.model2 = LlamaForCausalLM.from_pretrained('/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct')
        self.model2.eval()

        # 加载 model1
        self.tokenizer_model1 = PreTrainedTokenizerFast.from_pretrained('/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct')
        self.model1 = LlamaForCausalLM.from_pretrained('/root/lqs/LLaMA-Factory-main/llama3_models/merged_models')
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
  


