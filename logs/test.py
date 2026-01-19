import json
from collections import Counter

counts = Counter()
total = 0
with_gold = 0

with open("logs/bandit_data.jsonl") as f:
    for line in f:

        total += 1
        if json.loads(line).get("gold_answer") is not None:
            with_gold += 1

        r = json.loads(line)

        gold = r.get("gold_answer")   # SAFE access
        if gold is None:
            continue

        counts[r["action_index"]] += 1

print("Total logs:", total)
print("Dataset-backed logs:", with_gold)
print("Action distribution (dataset-backed only):")
print(counts)
