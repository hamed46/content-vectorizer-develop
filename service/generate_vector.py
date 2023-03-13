import json

from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("dangvantuan/sentence-camembert-large")


def get_embeddings(data):
    embeddings = model.encode(data['description'])
    print(embeddings)
    json_data = json.dumps(embeddings.tolist())
    return (json_data)
