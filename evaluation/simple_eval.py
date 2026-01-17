from retrieval.retriever import Retriever
from models.generator import Generator
from models.model_registry import ModelRegistry
from monitoring.metric import CostModel, Timer
from retrieval_policy import AdaptiveRetrievalPolicy
from policy.model_policy import AdaptiveModelPolicy

queries = [
    "What is machine learning?",
    "What is retrieval augmented generation?",
    "Explain FAISS indexing",
    "Summarize the causes of World War I"
]

retriever = Retriever(
    "data/index.faiss", "data/documents.json", k=10
)
# generator = Generator()
models = ModelRegistry()
policy = AdaptiveRetrievalPolicy()
model_policy = AdaptiveModelPolicy()

# cost_model = CostModel()
for q in queries:
    with Timer() as timer:
        k = policy.select_k(q)
        ctx = retriever.retrieve(q) if k > 0 else []
        model_id = model_policy.select_model(q, k)
        generator = models.get(model_id)
        ans, tokens = generator.generate(q, ctx)
    
    cost = tokens["total_tokens"] * models.cost_factor(model_id)

    print("=" * 70)
    print("Q:", q)
    print("k:", k, "| model:", model_id)
    print("Latency:", round(timer.elapsed, 2))
    print("Tokens:", tokens)
    print("Normalized cost:", cost)
    print("Answer preview:", ans)
