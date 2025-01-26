from sentence_transformers import SentenceTransformer

class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = None

    def calc_embeddings(self, text: str):
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
