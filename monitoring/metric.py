import time
from dataclasses import dataclass

@dataclass
class QueryMetrics:
    prompt_token: int
    completion_tokens: int
    total_tokens: int
    latency_sec: float
    normalized_cost: float

class CostModel:
    def __init__(self, cost_per_token: float = 1.0):
        self.cost_per_token = cost_per_token
    
    def compute_cost(self, total_tokens: int) -> float:
        return total_tokens * self.cost_per_token

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.elapsed = self.end - self.start