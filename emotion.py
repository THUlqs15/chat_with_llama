import openai

# 当前状态
current_state = (x, y)  # x, y 在 [1, 10] 范围内
user_input = "今天感觉心情特别低落，没有动力做事情。"
response = "我明白这种感觉，有时候我们都需要休息一下，放松自己的心情。你可以考虑做一些你喜欢的事情放松一下。"

# 准备输入给大模型的 prompt
prompt = f"""
当前情感状态：(开心度={current_state[0]}, 满意度={current_state[1]})
用户输入："{user_input}"
Chatbot的回复："{response}"

请根据用户输入和Chatbot的回复，决定情感状态应该如何更新。
你可以在以下方向中选择一个组合：
- 横向移动：[-1, 0, +1]
- 纵向移动：[-1, 0, +1]

请返回推荐的横向和纵向的移动值，分别在[-1, 0, +1]范围内。
例如：横向移动=+1, 纵向移动=-1
"""

# 调用 OpenAI API 获取状态更新建议
update_response = openai.Completion.create(
    model="gpt-3.5-turbo",
    prompt=prompt,
    max_tokens=50,
    temperature=0.5
)

# 解析返回结果，获得状态更新方向
update_text = update_response['choices'][0]['text'].strip().lower()
print(f"模型返回的更新建议: {update_text}")

# 提取横向和纵向的移动值
try:
    # 假设返回格式为 "横向移动=+1, 纵向移动=-1"
    horizontal_move = int(update_text.split("横向移动=")[1].split(",")[0].strip())
    vertical_move = int(update_text.split("纵向移动=")[1].strip())
except Exception as e:
    print(f"解析出错: {e}")
    horizontal_move = 0
    vertical_move = 0

# 根据更新方向更新状态，确保不越界
new_x = current_state[0] + horizontal_move
new_y = current_state[1] + vertical_move

# 限制范围在 [1, 10]
new_x = max(1, min(10, new_x))
new_y = max(1, min(10, new_y))

# 更新后的状态
current_state = (new_x, new_y)

# 打印更新后的状态
print(f"更新后的状态：(开心度={current_state[0]}, 满意度={current_state[1]})")

