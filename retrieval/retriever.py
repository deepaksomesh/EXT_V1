import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path, doc_path, graph_path="data/knowledge_graph.gml", k=5):
        self.k = k
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Graph Retriever (Optional)
        from retrieval.graph_retriever import GraphRetriever
        self.graph_retriever = GraphRetriever(graph_path)

        with open(doc_path, "r") as f:
            self.docs = json.load(f)

    def retrieve(self, query: str, use_graph: bool = False):
        # 1. Dense Retrieval
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

        dense_results = results[:self.k]
        
        # 2. Graph Retrieval (if enabled)
        if use_graph:
            graph_results = self.graph_retriever.retrieve(query)
            # Simple hybrid fusion: Append graph results to dense results
            # In production, use Reciprocal Rank Fusion (RRF)
            for gr in graph_results:
                if gr not in seen:
                    dense_results.append(f"[Graph] {gr}")
                    seen.add(gr)
        
        return dense_results

