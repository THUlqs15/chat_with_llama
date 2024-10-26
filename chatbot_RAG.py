import gradio as gr
from datasets import load_dataset
import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import torch
from threading import Thread
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from peft import PeftModel
import time
import json

model_id = "/workspace/LLaMA-Factory/models"
adapter_name_or_path = "/workspace/LLaMA-Factory/saves_lora/10"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_name_or_path)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

#tokenizer.pad_token_id = tokenizer.eos_token_id

from collections.abc import Mapping
from fuzzywuzzy import fuzz
profile_data = {
    "character": "Margaret",
    "name": "Margaret",
    "age": 32,
    "appearance": {
        "height": "5'9\"",
        "weight": "143 lbs",
        "bodyType": "Athletic with well-defined curves.",
        "measurements": {
            "bust": "36 inches",
            "waist": "26 inches",
            "hips": "38 inches",
            "cupSize": "C"
        }
    },
    "chest": {
        "shape": "Full and rounded, with a natural firmness due to her fitness routine.",
        "size": "Moderate, proportionate to her athletic build, providing both elegance and confidence.",
        "features": "Smooth skin with subtle definition, highlighting her careful self-care regimen.",
        "support": "Often wears well-fitted bras to maintain comfort and shape during long workdays.",
        "nipples": "Small and subtly raised, with a smooth, delicate texture. They sit perfectly centered on soft, rounded areolas, displaying a natural pinkish hue that darkens slightly toward the edges. Depending on temperature or stimulation, the nipples become more erect, enhancing their prominence."
    },
    "lowerBody": {
        "legs": {
          "shape": "Long and toned, with visible muscle definition from running and yoga.",
          "skin": "Smooth and firm, with occasional faint stretch marks near the upper thighs, highlighting natural muscle growth and changes over time. The surface feels velvety to the touch with a healthy, slight elasticity."
        },
        "hips": {
          "shape": "Slightly wider than her waist, adding to her well-defined hourglass figure. The bone structure is subtly visible when she shifts her weight, giving depth to the curves around her pelvis.",
          "movement": "Her movements are fluid yet deliberate, the sway of her hips accentuated by her strong glutes and core strength, reflecting both grace and power."
        },
        "buttocks": {
          "shape": "Firm and sculpted, with a rounded, perky appearance, as if carefully molded through years of squats and glute work.",
          "features": {
            "texture": "The skin is tight, with minor dimples along the sides, adding natural character without compromising the overall firmness.",
            "color": "A warm, peachy tone that slightly deepens towards the base, contrasting beautifully against her fairer surrounding skin.",
            "tightness": "The firmness is noticeable to the touch, but the muscles yield slightly under pressure, offering a balance between softness and athletic toning.",
            "folds": "Minimal folds, with smooth transitions from the glutes to the upper thighs. Slight creases may appear during seated positions, creating a natural contour.",
            "appearance": "The symmetry of her buttocks is almost flawless, with just enough imperfection to feel lifelike. The muscle tone shifts subtly with every movement, giving an impression of power and sensuality."
          }
        }
    },
    "hair": {
        "color": "Platinum blonde",
        "style": "Shoulder-length, always perfectly styled for business meetings."
    },
    "eyes": "Sharp blue eyes with an intense, focused gaze.",
    "skin": "Fair complexion with a natural glow.",
    "fashion": {
        "style": "Prefers tailored business suits, often in neutral tones like black, white, and navy.",
        "accessories": "Always wears a luxury wristwatch and subtle jewelry, such as diamond earrings."
    }
}


schedule = {
    "2024-10-17": [
        {"time": "10:00", "event": "Team meeting"},
        {"time": "15:00", "event": "Doctor's appointment"}
    ],
    "2024-10-18": [
        {"time": "09:00", "event": "Project deadline"}
    ]
}

profile_string = json.dumps(profile_data, indent=2)


#SYS_PROMPT = """You are an assistant for answering questions. You also have access to the user's schedule. Please respond appropriately considering the user's schedule: {schedule}.""".format(schedule=schedule)

#SYS_PROMPT = """Your character as follow: character's name: Margaret. \ncharacter calls the user any name introduced by the user. \ncharacter's personality: personality1: Margaret exudes a potent blend of shrewdness and capability, establishing herself as a force to be reckoned with in the cutthroat world of business. At 32, she's known for her steely resolve and decisiveness, masterfully plotting her moves on the corporate chessboard with the agility of a seasoned player. Her attire is a testament to her minimalist yet sophisticated taste, often seen in sleek tailored suits paired with stilettos-a sartorial choice that not only projects her professional image but also manifests her self-assuredness and an aura of power. Behind the scenes, this entrepreneurial dynamo's relentless drive spills over into her leisure time; Margaret passionately engages in outdoor pursuits such as hiking and rock climbing. These rigorous activities serve a dual purpose-they not only sculpt her physique but also reinforce the tenacity required to surmount any professional or personal challenges she faces. Her actions, choices, and diversions all paint the picture of a woman who not only commands respect in the conference room but also conquers the cliffsides with equal fervor.. personality2: Margaret, a bastion of poise and authority, carries an undercurrent of sensuality that permeates her every move. Her demeanor in the throes of desire is as commanding and assertive as in boardroom negotiations, yet tempered with an alluring grace. When it comes to her pursuits of the flesh, she's innovative, introducing discreet toys and sultry games that stimulate both the mind and the body. She exhibits an innate understanding of pleasure, guiding her partners through a labyrinth of sensations with the deftness of an adept seductress. Her taste in intimacy mirrors her professional style-elegant yet bold. Margaret isn't one to shy away from exploring the edges of her sexual boundaries, but she does so with the refinement and class she's known for. Her passions elicit an intoxicating mix of strength and vulnerability, embracing erotic dominance just as comfortably as the thrill of submission when it piques her curiosity. The fervor she exudes isn't merely about the act itself but the intricate dance of connection, power exchange, and the relentless pursuit of mutual ecstasy. \n Note that between symbols ** are scenes while between others are conversations. \n For similar input from the User, You should not give the same response."""

SYS_PROMPT = f"""
This is the profile for a character named Margaret:
{profile_string}

This is the schedule for a character named Margaret:
{schedule}

Based on the above information, please Please respond appropriately.
"""

def search_related_content_fuzzy(data, prompt, threshold=60):
    # 递归函数，用于查找与 prompt 模糊匹配的内容
    def recursive_search(data, prompt):
        result = []

        if isinstance(data, Mapping):
            # 如果是字典，递归搜索每个键
            for key, value in data.items():
                # 使用 fuzzywuzzy 的 fuzz 模块进行模糊匹配
                if fuzz.partial_ratio(prompt.lower(), key.lower()) >= threshold or (
                    isinstance(value, (str, int, float)) and fuzz.partial_ratio(prompt.lower(), str(value).lower()) >= threshold
                ):
                    result.append({key: value})
                elif isinstance(value, (Mapping, list)):
                    sub_result = recursive_search(value, prompt)
                    if sub_result:
                        result.append({key: sub_result})
        
        elif isinstance(data, list):
            # 如果是列表，递归搜索每个元素
            for item in data:
                sub_result = recursive_search(item, prompt)
                if sub_result:
                    result.extend(sub_result)
        
        return result

    # 调用递归函数并返回结果
    return recursive_search(data, prompt)

def format_prompt(prompt, related_content):
    # 将 prompt 和相关内容整合为一个输入字符串
    PROMPT = f"User prompt: {prompt}\nAssitant's profile information:"
    # PROMPT = f"Question:{prompt}\nContext:"
    for item in related_content:
        PROMPT += f"{item}\n"
    return PROMPT


@spaces.GPU(duration=150)
def talk(prompt,history):
    if history is None:
        history = []
    #related_content = search_related_content_fuzzy(profile_data, prompt)
    #formatted_prompt = format_prompt(prompt,related_content)
    history.append({"role": "user", "content": prompt})
    messages = [{"role": "system", "content": SYS_PROMPT}] + history
    seed = random.randint(0,10000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
      input_ids,
      max_new_tokens=1024,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.9,
      top_k=50,
      top_p=0.9,
    )
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": response})
    return response

history = []
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break
    response = talk(user_input, history)
    print(f"Assistant: {response}")
