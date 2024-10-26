#from datasets import load_dataset
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
import os
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
dataset = load_dataset("not-lain/wikipedia",revision = "embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings")
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
    }
}


def search(query: str, k: int = 3 ):
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples(
        "embeddings", embedded_query, 
        k=k
    )
    return scores, retrieved_examples




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

#scores , result = search("Jinping Xi", 3) 

prompt = "I like your chest."  # 示例提示词
related_content = search_related_content_fuzzy(profile_data, prompt)
print(related_content)

profile_data_list = {k: [v] for k, v in profile_data.items()}
profile_dataset = Dataset.from_dict(profile_data_list)




