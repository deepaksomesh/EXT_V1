from fastapi import FastAPI
from pydantic import BaseModel
import random

from retrieval.retriever import Retriever
from models.model_registry import ModelRegistry
from monitoring.metric import Timer
from policy.linucb_policy import LinUCBPolicy
from policy.data_logger import DataLogger
from data.qa_loader import QALoader

app = FastAPI()

# ------------------------
# Core components
# ------------------------
retriever = Retriever(
    index_path="data/index.faiss",
    doc_path="data/processed/documents.json",
    graph_path="data/knowledge_graph.gml",
    k=5
)

models = ModelRegistry()
# UPGRADE: Switched to LinUCB Policy
policy = LinUCBPolicy(alpha=1.0)
logger = DataLogger()
qa_loader = QALoader()

# ------------------------
# Request schema
# ------------------------
class QueryRequest(BaseModel):
    query: str | None = None
    use_dataset: bool = False
    use_graph: bool = True  # New flag for GraphRAG


@app.post("/query")
def query_rag(req: QueryRequest):
    """
    Phase 5.5:
    - Supports dataset-backed queries (training/eval)
    - Supports free-form queries (demo)
    - Logs everything needed for offline learning
    """

    # ------------------------
    # Query source
    # ------------------------
    if req.use_dataset:
        qa = qa_loader.sample()
        query = qa["question"]
        gold_answer = qa["answer"]
    else:
        query = req.query
        gold_answer = None

    # ------------------------
    # Joint decision policy
    # ------------------------
    action = policy.select_action(query)
    k = action["k"]
    model_id = action["model"]

    # ------------------------
    # RAG execution
    # ------------------------
    with Timer() as timer:
        retriever.k = k
        # Hybrid retrieval (Dense + Graph)
        contexts = retriever.retrieve(query, use_graph=req.use_graph)


        generator = models.get(model_id)
        answer, tokens = generator.generate(query, contexts)

    latency = timer.elapsed
    cost = tokens["total_tokens"] * models.cost_factor(model_id)

    # ------------------------
    # Logging for Phase 6
    # ------------------------
    logger.log({
        "query": query,
        "gold_answer": gold_answer,
        "generated_answer": answer,
        "retrieval_k": k,
        "model": model_id,
        "action_index": policy.ACTIONS.index(action),
        "contexts": contexts,
        "cost": cost,
        "latency": latency,
        "tokens": tokens
    })

    # ------------------------
    # Response
    # ------------------------
    return {
        "query": query,
        "action": action,
        "answer": answer,
        "metrics": {
            "latency_sec": round(latency, 2),
            "cost": round(cost, 2),
            "total_tokens": tokens["total_tokens"]
        },
        "gold_answer": gold_answer  # returned ONLY for inspection
    }
