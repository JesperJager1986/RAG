import numpy as np
from sentence_transformers import SentenceTransformer

class Model:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = None

    def __call__(self, text: str) -> np.ndarray:
        embedding: np.ndarray = self.model.encode(text, convert_to_numpy=True)
        return embedding

