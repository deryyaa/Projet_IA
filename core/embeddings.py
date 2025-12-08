from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialise le modèle SBERT.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        """
        Prend une liste de phrases et renvoie leurs embeddings.
        """
        return self.model.encode(sentences, convert_to_tensor=True)
