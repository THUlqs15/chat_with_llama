import requests

class ChatAgent:
    def __init__(self):
        self.history = []
        self.current_model = 'model2'  # 初始使用标准模型model2

    def call_model_api(self, prompt):
        # 首先调用 model2
        response = self.call_model2_api(history, prompt)
        
        if "cannot engage in" in response:
            # 切换到 model1
            if self.current_model != 'model1':
                self.current_model = 'model1'
            # 调用 model1
            response = self.call_model1_api(history, prompt)
        else:
            if self.current_model != 'model2':
                self.current_model = 'model2'

        return response

    def call_model1_api(self, prompt):
        # 调用 model1 的 API（需替换为实际的 API 调用）
        data = {
            'prompt': prompt,
            'history': self.history
        }
        # 示例：response = requests.post('https://api.model1.com/generate', json=data)
        response = {'text': f"来自 model1 的回复：'{prompt}'"}  # 模拟回复
        return response['text']

        

    def call_model2_api(self, prompt):
        data = {
            'prompt': prompt,
            'history': self.history
        }
        # 示例：response = requests.post('https://api.model2.com/generate', json=data)
        response_text = f"来自 model2 的回复：'{prompt}'"
        response = {'text': response_text}  # 模拟回复
        return response['text']

    def chat(self, prompt):
        response = self.call_model_api(prompt)
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
