import time
from fastapi import FastAPI
from pydantic import BaseModel

from retrieval.retriever import Retriever
from models.generator import Generator
from models.model_registry import ModelRegistry
from monitoring.metric import CostModel, Timer
from retrieval_policy import AdaptiveRetrievalPolicy
from policy.model_policy import AdaptiveModelPolicy
from policy.joint_policy import JointDecisionPolicy
from policy.data_logger import DataLogger

app = FastAPI()

retriever = Retriever(
    index_path="data/index.faiss",
    doc_path="data/documents.json",
    k=5
)

models = ModelRegistry()
# generator = Generator()
# cost_model = CostModel(cost_per_token=1.0)
policy = JointDecisionPolicy()
model_policy = AdaptiveModelPolicy()
logger = DataLogger()
cost_model = CostModel()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(req: QueryRequest):
    with Timer() as timer:
        action = policy.select_action(req.query)
        k = action["k"]
        model_id = action["model"]

        retriever.k = k
        contexts = retriever.retrieve(req.query)

        generator = models.get(model_id)
        answer, tokens = generator.generate(req.query, contexts)

    latency = timer.elapsed
    cost = tokens["total_tokens"] * models.cost_factor(model_id)

    # crude quality proxy: context used + output length
    quality_proxy = len(contexts) + tokens["completion_tokens"] / 40

    logger.log({
        "query": req.query,
        "action_index": policy.ACTIONS.index(action),
        "cost": cost,
        "latency": latency,
        "quality_proxy": quality_proxy
    })

    return {
        "query": req.query,
        "action": action,
        "answer": answer,
        "metrics": {
            "latency": round(latency, 2),
            "cost": cost
        }
    }