import json
import random

class QALoader:
    """
    Loads QA pairs and provides random sampling.
    Used for dataset-backed queries (Phase 5.5+).
    """

    def __init__(self, path="data/processed/qa_pairs.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.qa_pairs = json.load(f)

        if not self.qa_pairs:
            raise ValueError("QA dataset is empty")

    def sample(self):
        return random.choice(self.qa_pairs)
