#from datasets import load_dataset
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
import os
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
dataset = load_dataset("not-lain/wikipedia",revision = "embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings")


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
    }
}


def search(query: str, k: int = 3 ):
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples(
        "embeddings", embedded_query, 
        k=k
    )
    return scores, retrieved_examples


#scores , result = search("Jinping Xi", 3) 

profile_data_list = [profile_data]
profile_dataset = Dataset.from_dict(profile_data_list)
print("ha")
print(profile_dataset[0])



