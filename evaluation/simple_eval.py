import json
import random
from monitoring.metric import Timer
from retrieval.retriever import Retriever
from models.model_registry import ModelRegistry
from policy.joint_policy import JointDecisionPolicy

# ------------------------
# Load QA dataset
# ------------------------
with open("data/processed/qa_pairs.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

sampled_qa = random.sample(qa_pairs, 5)

# ------------------------
# Components
# ------------------------
retriever = Retriever(
    index_path="data/index.faiss",
    doc_path="data/processed/documents.json",
    k=5
)

models = ModelRegistry()
policy = JointDecisionPolicy()

# ------------------------
# Evaluation loop
# ------------------------
for qa in sampled_qa:
    query = qa["question"]
    gold = qa["answer"]

    action = policy.select_action(query)
    k = action["k"]
    model_id = action["model"]

    with Timer() as timer:
        retriever.k = k
        contexts = retriever.retrieve(query)
        generator = models.get(model_id)
        answer, tokens = generator.generate(query, contexts)

    cost = tokens["total_tokens"] * models.cost_factor(model_id)

    print("=" * 80)
    print("Q:", query)
    print("Gold answer:", gold)
    print("Action:", action)
    print("Latency (s):", round(timer.elapsed, 2))
    print("Cost:", round(cost, 2))
    print("Retrieved docs:", len(contexts))
    print("Generated answer preview:")
    print(answer[:300])
