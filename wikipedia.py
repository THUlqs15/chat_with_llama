from datasets import load_dataset
from sentence_transformers import SentenceTransformer
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")


dataset.push_to_hub("not-lain/wikipedia", revision="embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings")

def search(query: str, k: int = 3 ):
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples(
        "embeddings", embedded_query, 
        k=k
    )
    return scores, retrieved_examples


scores , result = search("Jinping Xi", 3) 
print(result['title'])
print(result['text'][0])
print(result['text'][1])
print(result['text'][2])


