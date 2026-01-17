import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

class JointDecisionPolicy:
    """
    Joint policy selecting (k, model_id).
    """

    ACTIONS = [
        {"k": 3, "model": "small"},
        {"k": 5, "model": "small"},
        {"k": 3, "model": "medium"},
        {"k": 5, "model": "medium"},
    ]

    def __init__(self):
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = LogisticRegression(max_iter=1000)
        self.trained = False

    def featurize(self, query: str):
        tokens = len(query.split())
        emb = self.embedder.encode(query)
        return np.concatenate([[tokens], emb[:10]])  # cheap + stable

    def select_action(self, query: str):
        x = self.featurize(query).reshape(1, -1)

        if not self.trained:
            return self.ACTIONS[0]  # safe default

        idx = self.model.predict(x)[0]
        return self.ACTIONS[idx]

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True
