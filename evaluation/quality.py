import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSimilarity:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, generated: str, gold: str) -> float:
        if not generated or not gold:
            return 0.0

        emb = self.model.encode([generated, gold])
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        return float(sim)
