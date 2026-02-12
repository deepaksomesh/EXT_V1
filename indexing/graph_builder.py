"""
GraphRAG Builder: Constructs a Knowledge Graph from text chunks.

Uses an LLM (Flan-T5) to extract (Subject, Relation, Object) triplets.
Stores the graph using NetworkX.
"""

import networkx as nx
import json
import re
import os
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class GraphBuilder:
    def __init__(self):
        # Using Flan-T5 for extraction
        self.model_name = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.graph = nx.MultiDiGraph()

    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract KG triplets from text using one-shot prompting.
        Returns list of (subject, relation, object).
        """
        prompt = (
            "Extract relationships from the text as (Subject, Relation, Object) triplets.\n"
            "Example:\n"
            "Text: Apple Inc. was founded by Steve Jobs in Cupertino.\n"
            "Output: (Apple Inc., founded by, Steve Jobs), (Apple Inc., location, Cupertino)\n\n"
            f"Text: {text}\n"
            "Output:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return self._parse_triplets(decoded)

    def _parse_triplets(self, raw_output: str) -> List[Tuple[str, str, str]]:
        """Regex parsing of LLM output."""
        matches = re.findall(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', raw_output)
        clean_triplets = []
        for s, r, o in matches:
            clean_triplets.append((s.strip(), r.strip(), o.strip()))
        return clean_triplets

    def build_from_documents(self, doc_path: str, save_path: str = "data/knowledge_graph.gml"):
        """Iterate over documents, extract triplets, and save graph."""
        if not os.path.exists(doc_path):
            print(f"[!] Document path {doc_path} not found.")
            return

        with open(doc_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        print(f"[*] Extracting graph from {len(docs)} documents...")
        
        # Limit processing for demo speed (first 10 docs)
        for i, doc in enumerate(docs[:10]):
            text = doc["text"]
            triplets = self.extract_triplets(text)
            
            for s, r, o in triplets:
                print(f"    + Edge: {s} --[{r}]--> {o}")
                self.graph.add_edge(s, o, relation=r, doc_id=i)
        
        nx.write_gml(self.graph, save_path)
        print(f"[*] Graph saved to {save_path} with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

if __name__ == "__main__":
    builder = GraphBuilder()
    builder.build_from_documents("data/processed/documents.json")
