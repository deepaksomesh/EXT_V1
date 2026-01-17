import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

class AdaptiveModelPolicy:
    """
    Chooses which LLM to use based on query difficulty.
    """

    def __init__(self):
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.classifier = LogisticRegression()
        self._initialize_dummy_policy()

    def _initialize_dummy_policy(self):
        X = np.array([
            [5, 0.5],
            [8, 0.9],
            [12, 1.5],
            [20, 2.5]
        ])
        y = np.array([0, 0, 1, 1])  # 0=small, 1=medium
        self.classifier.fit(X, y)

    def extract_features(self, query: str, retrieval_k: int):
        tokens = len(query.split())
        emb_norm = np.linalg.norm(self.embedder.encode(query))
        return np.array([tokens + retrieval_k, emb_norm])

    def select_model(self, query: str, retrieval_k: int) -> str:
        features = self.extract_features(query, retrieval_k).reshape(1, -1)
        idx = self.classifier.predict(features)[0]
        return "small" if idx == 0 else "medium"
