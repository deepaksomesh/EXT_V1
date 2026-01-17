import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path, doc_path, k=5):
        self.k = k
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        with open(doc_path, "r") as f:
            self.docs = json.load(f)

    def retrieve(self, query: str):
        query_emb = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_emb), self.k
        )

        results = []
        seen = set()

        for idx in indices[0]:
            text = self.docs[idx]["text"]
            if text not in seen:
                seen.add(text)
                results.append(text)

        return results[:self.k]
