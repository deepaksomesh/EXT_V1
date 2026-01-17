import json
from datetime import datetime

class DataLogger:
    def __init__(self, path="logs/bandit_data.jsonl"):
        self.path = path

    def log(self, record: dict):
        record["timestamp"] = datetime.utcnow().isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
