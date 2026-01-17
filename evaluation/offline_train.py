import json
import numpy as np
from policy.joint_policy import JointDecisionPolicy

LOG_PATH = "logs/bandit_data.jsonl"

def compute_reward(record):
    """
    Simple reward:
    higher quality proxy, lower cost & latency
    """
    quality = record["quality_proxy"]
    cost = record["cost"]
    latency = record["latency"]

    return quality - 0.01 * cost - 0.1 * latency


def main():
    policy = JointDecisionPolicy()
    X, y = [], []

    with open(LOG_PATH) as f:
        for line in f:
            r = json.loads(line)
            x = policy.featurize(r["query"])

            reward = compute_reward(r)
            action_idx = r["action_index"]

            # Only keep good outcomes
            if reward > 0:
                X.append(x)
                y.append(action_idx)

    X = np.array(X)
    y = np.array(y)

    if len(X) > 10:
        policy.train(X, y)
        print("Joint policy trained on", len(X), "samples")
    else:
        print("Not enough data yet")

if __name__ == "__main__":
    main()
