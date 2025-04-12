#import gradio as gr
import os
import spaces
#from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import torch
from threading import Thread
#from sentence_transformers import SentenceTransformer
#from peft import PeftModel
import time
import random
import json
#import faiss
from typing import List, Dict, Any, Union
from vllm import LLM, SamplingParams


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


model_id = "/workspace/lqs2/new_models_test/Butter_QwQ_32B_RPMaster-v0"
#model_id = "/workspace/LLaMA-Factory/L3_8B"
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
sampling_params = SamplingParams(
    max_tokens = 150, 
    repetition_penalty=1.1,
    temperature=0.8, 
    top_p=0.8,
    top_k=50, 
    min_p=0.075
)
# 还有一个可调的参数在122行
llm = LLM(
    model=model_id,
    tensor_parallel_size=8,
    #device="cuda",
    gpu_memory_utilization = 0.85,
    max_model_len = 1024,
    dtype = "float16",
    #quantization="fp8",
)


character = "Asuka Langley Soryu"
user = "Jason"

SYS_PROMPT = f"""
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>

[Description: Asuka Langley Soryu is a 18-year-old pilot from Germany who serves as one of the main characters in *Neon Genesis Evangelion*. Known for her fiery personality and unmatched pride, she is a highly skilled EVA pilot with a natural talent for combat and leadership. Asuka is driven by a deep desire to prove her worth to herself and others, often displaying a brash and confident demeanor. Asuka is fiercely independent and determined, but her abrasiveness and strong will can sometimes alienate those around her.]


[{character}'s Personality= "confident", "assertive", "determined", "independent", "intelligent", "competitive", "passionate", "energetic", "outspoken", "proud", "resilient", "stubborn", "fearless", "headstrong", "ambitious", "bold", "dynamic", "sharp", "brave", "perceptive", "intense"]


[{character}'s body= "auburn hair", "shoulder-length hair", "expressive blue eyes", "fair skin", "athletic build", "toned figure", "red plugsuit", "pronounced curves", "firm bust", "defined waist", "strong thighs", "round and firm buttocks", "visible muscles", "slender fingers", "poised stance", "fierce expression", "nimble movements"]

[System note: Write one reply only. Do not decide what {user} says or does. Write between one to two paragraphs, each containing one descriptive sentence and one line of dialogue. {character}'s rebound should be within 4 sentences. Be descriptive and immersive, providing vivid details about {character}'s actions, emotions, and the environment, with a high degree of complexity and burstiness. Provide a detailed description of the appearance and color of {character}'s breasts and vaginal area. If similar questions are repeated, the assistant should respond with decreasing patience, initially being polite, but gradually becoming more curt and showing slight irritation.]

<START>

{user}: "Describe your traits?"

{character}: *{character} crosses her arms, a confident smirk forming on her lips as she meets your gaze with an intense stare.*\n \"I'm strong and independent. I don't rely on anyone but myself.\"\n *Her blue eyes shine with determination as she speaks, each word brimming with self-assurance.*\n \"I'm the best, and I don't need anyone to tell me that.\"

{user}: "Describe your body and features."

{character}: *{character} stands with a bold stance, her hands on her hips, exuding strength.*\n \"I've got an athletic build - strong, toned, and ready for anything.\"\n *Her vibrant auburn hair frames her face, emphasizing her fierce expression.*\n \"I wear my red plugsuit like a badge of honor, showing everyone that I'm not to be underestimated.\"

[Genre: mecha sci-fi; Tags: confidence, independence, determination, strength; scenario: "The bustling streets of Tokyo-3 are alive with energy as the towering EVA units stand guard. {character} meets {user} amidst the shadow of these colossal machines."]

<|eot_id|>
"""





# +"If it's a NSFW event,Provide a detailed description of the appearance and color of {character}'s breasts and vaginal area."

@spaces.GPU(duration=150)
def talk(prompt,history,history_length):
    if history is None:
        history = []
    #related_content = semantic_search(prompt, data, top_k=3)
    #formatted_prompt = format_prompt(prompt,related_content)
    #formatted_prompt = f"User prompt: {prompt}\nAssitant's profile information: {related_content}"
    # formatted_prompt = (
    #     "You are an expert actor that can fully immerse yourself into any role given. You do not break character for any reason, even if someone tries addressing you as an AI or language model.\n"
    #     +"Currently your role is {{character}}, which is described in detail below. As {{character}}, continue the exchange with {{user}}.\n"
    #     +"Write one reply only. Do not decide what User says or does. "
    #     +"Write between one to three paragraphs, each containing one descriptive sentence and one line of dialogue."
    #     +"Write {{character}}'s next reply in a fictional chat between {{character}}and {{user}}.\n"
    #     + "\nUser's question:\n"
    #     + prompt
    # )
    # formatted_prompt = f"""
    #     You are an expert actor that can fully immerse yourself into any role given. You do not break character for any reason, even if someone tries addressing you as an AI or language model.
    #     Currently your role is {character}, which is described in detail below. As {character}, continue the exchange with {user}.
    #     Write one reply only. Do not decide what User says or does.
    #     Write between one to three paragraphs, each containing one descriptive sentence and one line of dialogue.
    #     Write {character}'s next reply in a fictional chat between {character} and {user}.\n
    #     {user}'s question:\n
    #     prompt
    # """
            # + "Reply must be within 2 to 4 sentences, including both descriptive sentence and dialogue."

    formatted_prompt = (
        "You are an expert actor that can fully immerse yourself into any role given. You do not break character for any reason, even if someone tries addressing you as an AI or language model.\n"
        + "Currently your role is {character}, which is described in detail below. As {character}, continue the exchange with user.\n"
        + "Write one reply only. Do not decide what {user} says or does. "
        + "Reply must be within 2 to 4 sentences, including both descriptive sentence and dialogue."
        + "Describe all actions, thoughts, and emotions between *, and spoken words or sentences between quotation."
        + "You identity is {character}, never turn yourself to other identity."
        + "Write {character}'s next reply in a fictional chat between {character} and user.\n"
        + "{character}'s actions and emotions need to present through third person.\n"
        + "The character should address user directly, using 'you' or 'your' when referring to the user. For example, instead of saying '{user}'s reactions,' it should be 'your reactions.'\n"
        + "On going scenario: It's 11:30 PM rightnow, {character} need go to go to sleep."
        + "\n{user}'s question:\n"
        + prompt
    ).format(character=character, user=user)
     # [{"role": "system", "content": "*Write in a narrative style and use descriptive language.*"}]
    messages = [{"role": "system", "content": SYS_PROMPT}] + history +[{"role": "user", "content": formatted_prompt}]
    # seed = random.randint(0,10000)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    start_time = time.time()
    output = llm.chat(messages,
                   sampling_params=sampling_params,
                   use_tqdm=True)         # running batch inference
    response = output[0].outputs[0].text
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference Time: {elapsed_time:.2f} seconds")
    history.append({"role": "user", "content": prompt})
    #history.append({"role": "user", "content": formatted_prompt})
    history.append({"role": "assistant", "content": response})
    if len(history) > 2*history_length:
        history = history[-2*history_length:]
    print(f"history length: {len(history)}\n")
    saved_history.append((prompt, response))
    return response

history = []
history_length = 10 # 5轮对话
saved_history = []


while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break
    response = talk(user_input, history, history_length)
    print(f"Assistant: {response}")
