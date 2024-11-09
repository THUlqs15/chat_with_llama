from fastapi import FastAPI, Request
from typing import Dict, Any

app = FastAPI()
user_histories: Dict[str, Any] = {}


# load model
# talk 函数实现


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data["user_id"]
    prompt = data["prompt"]

    if user_id not in user_histories:
        user_histories[user_id] = []

    response = talk(prompt, user_histories[user_id], history_length=10)

    user_histories[user_id].append({"role": "user", "content": prompt})
    user_histories[user_id].append({"role": "assistant", "content": response})

    # 限制历史记录
    if len(user_histories[user_id]) > 20:
        user_histories[user_id] = user_histories[user_id][-20:]

    return {"response": response}
