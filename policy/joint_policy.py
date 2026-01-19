import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import random

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

    def select_action(self, query, epsilon: float = 0.2):
        """
        ε-greedy action selection for exploration.
        """

        # Exploration
        if random.random() < epsilon:
            return random.choice(self.ACTIONS)

        # Exploitation
        features = self.featurize(query).reshape(1, -1)
        action_idx = self.model.predict(features)[0]
        return self.ACTIONS[action_idx]


    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True
