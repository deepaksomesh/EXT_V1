"""
Policy Comparison: Fixed vs Adaptive

Compares the performance of:
1. Fixed policy (always k=5, always medium model)
2. Adaptive policy (JointDecisionPolicy with ε-greedy selection)

Outputs a comparison table with metrics.
"""

import json
import random
from tabulate import tabulate
from dataclasses import dataclass
from typing import List, Dict

from retrieval.retriever import Retriever
from models.model_registry import ModelRegistry
from policy.joint_policy import JointDecisionPolicy
from monitoring.metric import Timer
from evaluation.metrics import EnhancedMetrics


@dataclass
class PolicyResult:
    name: str
    answers: List[str]
    golds: List[str]
    latencies: List[float]
    costs: List[float]


def load_qa_pairs(path: str = "data/processed/qa_pairs.json", n_samples: int = 20) -> List[Dict]:
    """Load QA pairs for evaluation."""
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return random.sample(pairs, min(n_samples, len(pairs)))


def run_fixed_policy(qa_pairs, retriever, models) -> PolicyResult:
    """Run fixed policy: always k=5, always medium model."""
    answers, golds, latencies, costs = [], [], [], []

    fixed_k = 5
    fixed_model = "medium"

    for qa in qa_pairs:
        query = qa["question"]
        gold = qa["answer"]

        with Timer() as timer:
            retriever.k = fixed_k
            contexts = retriever.retrieve(query)
            generator = models.get(fixed_model)
            answer, tokens = generator.generate(query, contexts)

        latency = timer.elapsed
        cost = tokens["total_tokens"] * models.cost_factor(fixed_model)

        answers.append(answer)
        golds.append(gold)
        latencies.append(latency)
        costs.append(cost)

    return PolicyResult("Fixed (k=5, medium)", answers, golds, latencies, costs)


def run_adaptive_policy(qa_pairs, retriever, models, policy) -> PolicyResult:
    """Run adaptive policy with JointDecisionPolicy."""
    answers, golds, latencies, costs = [], [], [], []

    for qa in qa_pairs:
        query = qa["question"]
        gold = qa["answer"]

        # Adaptive selection
        action = policy.select_action(query, epsilon=0.0)  # Pure exploitation for evaluation
        k = action["k"]
        model_id = action["model"]

        with Timer() as timer:
            retriever.k = k
            contexts = retriever.retrieve(query)
            generator = models.get(model_id)
            answer, tokens = generator.generate(query, contexts)

        latency = timer.elapsed
        cost = tokens["total_tokens"] * models.cost_factor(model_id)

        answers.append(answer)
        golds.append(gold)
        latencies.append(latency)
        costs.append(cost)

    return PolicyResult("Adaptive (learned)", answers, golds, latencies, costs)


def compute_metrics(result: PolicyResult, metrics: EnhancedMetrics) -> Dict:
    """Compute all metrics for a policy result."""
    eval_results = metrics.evaluate_batch(result.answers, result.golds)

    return {
        "Policy": result.name,
        "Semantic Sim": eval_results["mean_semantic_similarity"],
        "Exact Match": eval_results["mean_exact_match"],
        "Token F1": eval_results["mean_token_f1"],
        "Precision": eval_results["mean_token_precision"],
        "Recall": eval_results["mean_token_recall"],
        "Avg Latency (s)": round(sum(result.latencies) / len(result.latencies), 3),
        "Avg Cost": round(sum(result.costs) / len(result.costs), 2),
        "Total Cost": round(sum(result.costs), 2),
        "Samples": eval_results["num_samples"]
    }


def main():
    print("[*] Loading components...")

    # Load components
    retriever = Retriever(
        index_path="data/index.faiss",
        doc_path="data/processed/documents.json",
        k=5
    )
    models = ModelRegistry()
    policy = JointDecisionPolicy()
    metrics = EnhancedMetrics()

    # Load QA pairs
    qa_pairs = load_qa_pairs(n_samples=15)
    print(f"[*] Evaluating on {len(qa_pairs)} samples...\n")

    # Run policies
    print("[>] Running Fixed Policy...")
    fixed_result = run_fixed_policy(qa_pairs, retriever, models)

    print("[>] Running Adaptive Policy...")
    adaptive_result = run_adaptive_policy(qa_pairs, retriever, models, policy)

    # Compute metrics
    fixed_metrics = compute_metrics(fixed_result, metrics)
    adaptive_metrics = compute_metrics(adaptive_result, metrics)

    # Display comparison table
    print("\n" + "=" * 100)
    print("POLICY COMPARISON RESULTS")
    print("=" * 100 + "\n")

    table_data = [fixed_metrics, adaptive_metrics]
    headers = list(fixed_metrics.keys())

    print(tabulate(table_data, headers="keys", tablefmt="grid", floatfmt=".4f"))

    # Summary
    print("\n" + "-" * 100)
    print("SUMMARY")
    print("-" * 100)

    cost_diff = fixed_metrics["Total Cost"] - adaptive_metrics["Total Cost"]
    cost_pct = (cost_diff / fixed_metrics["Total Cost"]) * 100 if fixed_metrics["Total Cost"] > 0 else 0

    sim_diff = adaptive_metrics["Semantic Sim"] - fixed_metrics["Semantic Sim"]

    print(f"* Cost Savings (Adaptive vs Fixed): {cost_diff:.2f} ({cost_pct:.1f}%)")
    print(f"* Semantic Similarity Difference: {sim_diff:+.4f}")

    if cost_diff > 0 and sim_diff >= -0.05:
        print("[OK] Adaptive policy saves cost with acceptable quality tradeoff!")
    elif sim_diff > 0.05:
        print("[OK] Adaptive policy achieves better quality!")
    else:
        print("[!] Results may need more training data for policy improvement.")


if __name__ == "__main__":
    main()
