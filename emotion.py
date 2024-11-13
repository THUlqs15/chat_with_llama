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

请根据用户输入和Chatbot的回复，决定情感状态应该如何更新。你可以在以下方向中选择一个：
- 向上 (增加满意度)
- 向下 (减少满意度)
- 向左 (减少开心度)
- 向右 (增加开心度)
- 保持不变

请提供推荐的更新方向，另外说明你选择此方向的原因。
"""

# 调用 OpenAI API 获取状态更新建议
update_response = openai.Completion.create(
    model="gpt-3.5-turbo",
    prompt=prompt,
    max_tokens=50,
    temperature=0.5
)

# 解析返回结果，获得状态更新方向
update_direction = update_response['choices'][0]['text'].strip().lower()

# 根据更新方向更新状态
if update_direction == "向上" and current_state[1] < 10:
    current_state = (current_state[0], current_state[1] + 1)
elif update_direction == "向下" and current_state[1] > 1:
    current_state = (current_state[0], current_state[1] - 1)
elif update_direction == "向左" and current_state[0] > 1:
    current_state = (current_state[0] - 1, current_state[1])
elif update_direction == "向右" and current_state[0] < 10:
    current_state = (current_state[0] + 1, current_state[1])
elif update_direction == "保持不变":
    pass  # 状态不变

# 打印更新后的状态
print(f"更新后的状态：(开心度={current_state[0]}, 满意度={current_state[1]})")




current_state = (x, y)
prefix = ""

if x > 5 and y > 5:
    prefix = "你现在非常开心和满意，尝试用一种热情愉快的语气来回答用户的问题："
elif x < 5 and y < 5:
    prefix = "你现在情绪有点低落，尝试用一种安慰和同情的语气来回答用户的问题："
# 其他状态的判断逻辑...

final_prompt = prefix + user_input
response = openai.Completion.create(
    model="gpt-3.5-turbo",
    prompt=final_prompt,
    max_tokens=100
)
