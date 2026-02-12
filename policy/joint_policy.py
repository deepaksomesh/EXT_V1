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
        self._initialize_with_prior()

    def _initialize_with_prior(self):
        """
        Initialize with a simple prior policy.
        Short queries -> small model, longer queries -> medium model.
        """
        # Create dummy features: [token_count, emb_dim_0..9]
        dummy_X = np.array([
            [3] + [0.1] * 10,   # Short query -> action 0 (k=3, small)
            [5] + [0.2] * 10,   # Short query -> action 1 (k=5, small)
            [10] + [0.5] * 10,  # Medium query -> action 2 (k=3, medium)
            [15] + [0.8] * 10,  # Long query -> action 3 (k=5, medium)
        ])
        dummy_y = np.array([0, 1, 2, 3])
        self.model.fit(dummy_X, dummy_y)

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

