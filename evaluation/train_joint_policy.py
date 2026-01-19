import json
import numpy as np
from policy.joint_policy import JointDecisionPolicy
from evaluation.quality import SemanticSimilarity
from evaluation.reward import compute_reward

LOG_PATH = "logs/bandit_data.jsonl"

def main():
    policy = JointDecisionPolicy()
    scorer = SemanticSimilarity()

    X = []
    y = []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            gold = record.get("gold_answer")
            generated = record.get("generated_answer")

            if gold is None:
                continue  # skip free-form queries

            similarity = scorer.score(generated, gold)
            reward = compute_reward(
                similarity=similarity,
                cost=record["cost"],
                latency=record["latency"]
            )

            features = policy.featurize(record["query"])

            X.append(features)
            y.append(record["action_index"])

    if len(X) < 20:
        print("❌ Not enough dataset-backed samples. Collect more logs.")
        return

    X = np.array(X)
    y = np.array(y)

    policy.train(X, y)

    print(f"✅ Joint policy trained on {len(X)} dataset-backed samples")

if __name__ == "__main__":
    main()
