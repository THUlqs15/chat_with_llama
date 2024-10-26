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
            }
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

print(profile_dataset[0])



