"""
Robust Policy Comparison with LLM-as-Judge

Enhanced comparison including:
1. Larger sample size (50+ samples)
2. Multiple evaluation runs for statistical reliability
3. LLM-as-judge scoring
4. Head-to-head comparison between policies
"""

import json
import random
from tabulate import tabulate
from dataclasses import dataclass
from typing import List, Dict
import statistics

from retrieval.retriever import Retriever
from models.model_registry import ModelRegistry
from policy.joint_policy import JointDecisionPolicy
from monitoring.metric import Timer
from evaluation.metrics import EnhancedMetrics
from evaluation.llm_judge import LLMJudge, ComparativeJudge


@dataclass
class PolicyResult:
    name: str
    answers: List[str]
    golds: List[str]
    questions: List[str]
    latencies: List[float]
    costs: List[float]


def load_qa_pairs(path: str = "data/processed/qa_pairs.json", n_samples: int = 50) -> List[Dict]:
    """Load QA pairs for evaluation."""
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    # Use all available if less than requested
    actual_n = min(n_samples, len(pairs))
    return random.sample(pairs, actual_n)


def run_fixed_policy(qa_pairs, retriever, models) -> PolicyResult:
    """Run fixed policy: always k=5, always medium model."""
    answers, golds, questions, latencies, costs = [], [], [], [], []

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
        questions.append(query)
        latencies.append(latency)
        costs.append(cost)

    return PolicyResult("Fixed (k=5, medium)", answers, golds, questions, latencies, costs)


def run_adaptive_policy(qa_pairs, retriever, models, policy) -> PolicyResult:
    """Run adaptive policy with JointDecisionPolicy."""
    answers, golds, questions, latencies, costs = [], [], [], [], []

    for qa in qa_pairs:
        query = qa["question"]
        gold = qa["answer"]

        # Adaptive selection - pure exploitation for evaluation
        action = policy.select_action(query, epsilon=0.0)
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
        questions.append(query)
        latencies.append(latency)
        costs.append(cost)

    return PolicyResult("Adaptive (learned)", answers, golds, questions, latencies, costs)


def compute_metrics(result: PolicyResult, metrics: EnhancedMetrics, judge: LLMJudge) -> Dict:
    """Compute all metrics including LLM judge scores."""
    # Standard metrics
    eval_results = metrics.evaluate_batch(result.answers, result.golds)
    
    # LLM Judge scoring
    judge_samples = list(zip(result.questions, result.golds, result.answers))
    judge_results = judge.judge_batch(judge_samples)

    return {
        "Policy": result.name,
        "Semantic Sim": eval_results["mean_semantic_similarity"],
        "Token F1": eval_results["mean_token_f1"],
        "LLM Score": judge_results["mean_score"],
        "Avg Latency": round(sum(result.latencies) / len(result.latencies), 2),
        "Avg Cost": round(sum(result.costs) / len(result.costs), 1),
        "Total Cost": round(sum(result.costs), 0),
        "Samples": eval_results["num_samples"]
    }


def run_head_to_head(fixed_result: PolicyResult, adaptive_result: PolicyResult, 
                     judge: ComparativeJudge) -> Dict:
    """Run head-to-head comparison between policies."""
    samples = []
    
    for i in range(len(fixed_result.questions)):
        samples.append((
            fixed_result.questions[i],
            fixed_result.golds[i],
            fixed_result.answers[i],  # Answer A = Fixed
            adaptive_result.answers[i]  # Answer B = Adaptive
        ))
    
    return judge.compare_batch(samples)


def run_multiple_evaluations(n_runs: int = 3, samples_per_run: int = 50):
    """Run multiple evaluation rounds and aggregate results."""
    
    print("[*] Loading components...")
    
    # Load components once
    retriever = Retriever(
        index_path="data/index.faiss",
        doc_path="data/processed/documents.json",
        k=5
    )
    models = ModelRegistry()
    policy = JointDecisionPolicy()
    metrics = EnhancedMetrics()
    judge = LLMJudge()
    comparative_judge = ComparativeJudge()
    
    all_fixed_scores = []
    all_adaptive_scores = []
    all_fixed_costs = []
    all_adaptive_costs = []
    all_head_to_head = {'A': 0, 'B': 0, 'Tie': 0}
    
    for run in range(n_runs):
        print(f"\n[>] Run {run + 1}/{n_runs}...")
        
        # Load fresh sample for each run
        qa_pairs = load_qa_pairs(n_samples=samples_per_run)
        
        # Run policies
        fixed_result = run_fixed_policy(qa_pairs, retriever, models)
        adaptive_result = run_adaptive_policy(qa_pairs, retriever, models, policy)
        
        # Get LLM judge scores
        fixed_samples = list(zip(fixed_result.questions, fixed_result.golds, fixed_result.answers))
        adaptive_samples = list(zip(adaptive_result.questions, adaptive_result.golds, adaptive_result.answers))
        
        fixed_judge = judge.judge_batch(fixed_samples)
        adaptive_judge = judge.judge_batch(adaptive_samples)
        
        all_fixed_scores.append(fixed_judge["mean_score"])
        all_adaptive_scores.append(adaptive_judge["mean_score"])
        all_fixed_costs.append(sum(fixed_result.costs))
        all_adaptive_costs.append(sum(adaptive_result.costs))
        
        # Head-to-head
        h2h = run_head_to_head(fixed_result, adaptive_result, comparative_judge)
        all_head_to_head['A'] += h2h['A_wins']
        all_head_to_head['B'] += h2h['B_wins']
        all_head_to_head['Tie'] += h2h['ties']
    
    return {
        "fixed": {
            "mean_llm_score": round(statistics.mean(all_fixed_scores), 3),
            "std_llm_score": round(statistics.stdev(all_fixed_scores), 3) if len(all_fixed_scores) > 1 else 0,
            "total_cost": round(sum(all_fixed_costs), 0)
        },
        "adaptive": {
            "mean_llm_score": round(statistics.mean(all_adaptive_scores), 3),
            "std_llm_score": round(statistics.stdev(all_adaptive_scores), 3) if len(all_adaptive_scores) > 1 else 0,
            "total_cost": round(sum(all_adaptive_costs), 0)
        },
        "head_to_head": all_head_to_head,
        "n_runs": n_runs,
        "samples_per_run": samples_per_run
    }


def main():
    print("=" * 100)
    print("ROBUST POLICY COMPARISON WITH LLM-AS-JUDGE")
    print("=" * 100)
    
    # Configuration
    N_RUNS = 3
    SAMPLES_PER_RUN = 30  # Adjust based on dataset size
    
    # Single detailed run first
    print("\n[*] Loading components for detailed analysis...")
    
    retriever = Retriever(
        index_path="data/index.faiss",
        doc_path="data/processed/documents.json",
        k=5
    )
    models = ModelRegistry()
    policy = JointDecisionPolicy()
    metrics = EnhancedMetrics()
    judge = LLMJudge()
    comparative_judge = ComparativeJudge()
    
    # Load larger sample
    qa_pairs = load_qa_pairs(n_samples=SAMPLES_PER_RUN)
    print(f"[*] Evaluating on {len(qa_pairs)} samples...\n")
    
    # Run policies
    print("[>] Running Fixed Policy...")
    fixed_result = run_fixed_policy(qa_pairs, retriever, models)
    
    print("[>] Running Adaptive Policy...")
    adaptive_result = run_adaptive_policy(qa_pairs, retriever, models, policy)
    
    # Compute metrics with LLM judge
    print("[>] Computing metrics with LLM-as-Judge...")
    fixed_metrics = compute_metrics(fixed_result, metrics, judge)
    adaptive_metrics = compute_metrics(adaptive_result, metrics, judge)
    
    # Display detailed comparison table
    print("\n" + "=" * 100)
    print("DETAILED RESULTS (Single Run)")
    print("=" * 100 + "\n")
    
    table_data = [fixed_metrics, adaptive_metrics]
    print(tabulate(table_data, headers="keys", tablefmt="grid", floatfmt=".3f"))
    
    # Head-to-head comparison
    print("\n[>] Running head-to-head comparison...")
    h2h_results = run_head_to_head(fixed_result, adaptive_result, comparative_judge)
    
    print("\n" + "-" * 100)
    print("HEAD-TO-HEAD COMPARISON (Fixed=A vs Adaptive=B)")
    print("-" * 100)
    print(f"Fixed Policy Wins:    {h2h_results['A_wins']} ({h2h_results['A_win_rate']*100:.1f}%)")
    print(f"Adaptive Policy Wins: {h2h_results['B_wins']} ({h2h_results['B_win_rate']*100:.1f}%)")
    print(f"Ties:                 {h2h_results['ties']} ({h2h_results['tie_rate']*100:.1f}%)")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    cost_diff = fixed_metrics["Total Cost"] - adaptive_metrics["Total Cost"]
    cost_pct = (cost_diff / fixed_metrics["Total Cost"]) * 100 if fixed_metrics["Total Cost"] > 0 else 0
    
    llm_diff = adaptive_metrics["LLM Score"] - fixed_metrics["LLM Score"]
    
    print(f"\n* Cost Savings (Adaptive vs Fixed): {cost_diff:.0f} ({cost_pct:.1f}%)")
    print(f"* LLM Judge Score Difference: {llm_diff:+.3f}")
    print(f"* Head-to-Head: Adaptive wins {h2h_results['B_win_rate']*100:.1f}% of comparisons")
    
    # Final verdict
    print("\n" + "-" * 100)
    if cost_pct > 50 and h2h_results['B_win_rate'] >= h2h_results['A_win_rate']:
        print("[OK] Adaptive policy provides significant cost savings with competitive quality!")
    elif h2h_results['B_win_rate'] > h2h_results['A_win_rate']:
        print("[OK] Adaptive policy achieves better quality overall!")
    elif cost_pct > 30:
        print("[OK] Adaptive policy provides meaningful cost savings!")
    else:
        print("[!] Results suggest need for more policy training or parameter tuning.")


if __name__ == "__main__":
    main()
