"""
GraphRAG Retriever using NetworkX.
Provides context by traversing entity relationships.
"""

import networkx as nx
import os
from typing import List

class GraphRetriever:
    def __init__(self, graph_path: str = "data/knowledge_graph.gml"):
        self.graph = nx.MultiDiGraph()
        if os.path.exists(graph_path):
            try:
                self.graph = nx.read_gml(graph_path)
                print(f"[*] Loaded Knowledge Graph with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                print(f"[!] Failed to load KG: {e}")
        else:
            print("[!] No Knowledge Graph found. Initialize with GraphBuilder.")

    def retrieve(self, query: str, hops: int = 1) -> List[str]:
        """
        Entity Linking + 1-hop expansion.
        """
        if self.graph.number_of_nodes() == 0:
            return []

        query_lower = query.lower()
        matched_nodes = []

        # Simple Entity Linking (String Match)
        for node in self.graph.nodes():
            if str(node).lower() in query_lower:
                matched_nodes.append(node)

        context = []
        visited = set()

        for start_node in matched_nodes:
            # Outgoing edges
            if start_node in self.graph:
                for neighbor in self.graph[start_node]:
                    edge_data = self.graph.get_edge_data(start_node, neighbor)
                    for key, data in edge_data.items():
                        rel = data.get('relation', 'related to')
                        sentence = f"{start_node} {rel} {neighbor}"
                        if sentence not in visited:
                            context.append(sentence)
                            visited.add(sentence)

        return context
