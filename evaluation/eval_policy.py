import json
import random
from policy.joint_policy import JointDecisionPolicy
from evaluation.quality import SemanticSimilarity

LOG_PATH = "logs/bandit_data.jsonl"

def main():
    scorer = SemanticSimilarity()

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if json.loads(line).get("gold_answer") is not None
        ]

    sample = random.sample(records, min(30, len(records)))

    similarities = []

    for r in sample:
        sim = scorer.score(r["generated_answer"], r["gold_answer"])
        similarities.append(sim)

    print("📊 Evaluation Results")
    print("Mean semantic similarity:", round(sum(similarities) / len(similarities), 3))
    print("Samples evaluated:", len(similarities))

if __name__ == "__main__":
    main()
