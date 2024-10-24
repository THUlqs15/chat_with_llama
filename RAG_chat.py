import gradio as gr
from datasets import load_dataset

import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import torch
from threading import Thread
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import time


ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

dataset = load_dataset("not-lain/wikipedia",revision = "embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings") # column name that has the embeddings of the dataset

model_id = "/workspace/LLaMA-Factory/models"
adapter_name_or_path = "/workspace/LLaMA-Factory/saves_lora/5"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
#model = PeftModel.from_pretrained(model, adapter_name_or_path)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

SYS_PROMPT = """You are an assistant for answering questions."""



def search(query: str, k: int = 3 ):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples( # retrieve results
        "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
        k=k # get only top k results
    )
    return scores, retrieved_examples

def format_prompt(prompt,retrieved_documents,k):
    """using the retrieved documents we will prompt the model to generate our responses"""
    PROMPT = f"Question:{prompt}\nContext:"
    for idx in range(k) :
        PROMPT+= f"{retrieved_documents['text'][idx]}\n"
    return PROMPT


@spaces.GPU(duration=150)
def talk(prompt,history):
    k = 1 # number of retrieved documents
    scores , retrieved_documents = search(prompt, k)
    formatted_prompt = format_prompt(prompt,retrieved_documents,k)
    formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
    messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
    # tell the model to generate
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
      temperature=0.6,
      top_p=0.9,
    )
    streamer = TextIteratorStreamer(
            tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        temperature=0.75,
        eos_token_id=terminators,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        print(outputs)
        yield "".join(outputs)


TITLE = "# RAG"

DESCRIPTION = """
A rag pipeline with a chatbot feature
Resources used to build this project :
* embedding model : https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
* dataset : https://huggingface.co/datasets/not-lain/wikipedia
* faiss docs : https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/main_classes#datasets.Dataset.add_faiss_index 
* chatbot : https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
* Full documentation : https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3 
"""


demo = gr.ChatInterface(
    fn=talk,
    chatbot=gr.Chatbot(
        show_label=True,
        show_share_button=True,
        show_copy_button=True,
        likeable=True,
        layout="bubble",
        bubble_full_width=False,
    ),
    theme="Soft",
    examples=[["what's anarchy ? "]],
    title=TITLE,
    description=DESCRIPTION,
    
)
demo.launch(debug=True)
