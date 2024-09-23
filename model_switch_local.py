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
      
    def call_model_api(self, prompt):
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
      
  
  


