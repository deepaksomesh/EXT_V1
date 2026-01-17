import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

class AdaptiveRetrievalPolicy:

    def __init__(self):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.classifier = LogisticRegression()
        self._initialize_dummy_policy()

        self.k_options = [0, 3, 5, 10]
    

    def _initialize_dummy_policy(self):

        X = np.array([
            [5, 0.2],
            [10, 0.8],
            [15, 1.5],
            [20, 2.5]
        ])

        y = np.array([0, 1, 2, 3])
        self.classifier.fit(X, y)
    
    def extract_features(self, query: str):
        tokens = len(query.split())
        embedding = self.embedder.encode(query)
        emb_norm = np.linalg.norm(embedding)
        return np.array([tokens, emb_norm])

    def select_k(self, query: str) -> int:
        features = self.extract_features(query).reshape(1, -1)
        idx = self.classifier.predict(features)[0]
        k = self.k_options[idx]

        # Strong safety rules
        query_lower = query.lower()
        if (
            len(query.split()) > 3 or
            query_lower.startswith("what") or
            query_lower.startswith("explain") or
            query_lower.startswith("summarize")
        ):
            k = max(k, 3)

        return k

