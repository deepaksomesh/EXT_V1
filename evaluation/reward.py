def compute_reward(
    similarity: float,
    cost: float,
    latency: float,
    lambda_cost: float = 0.01,
    lambda_latency: float = 0.1,
) -> float:
    """
    Final reward used for offline learning.
    """
    return similarity - lambda_cost * cost - lambda_latency * latency
