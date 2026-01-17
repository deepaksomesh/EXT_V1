from models.generator import Generator

class ModelRegistry:
    """
    Registry for multiple LLMs with cost coefficients.
    """

    def __init__(self):
        self.models = {
            "small": {
                "generator": Generator(model_name="distilgpt2"),
                "cost_per_token": 0.5
            },
            "medium": {
                "generator": Generator(model_name="gpt2"),
                "cost_per_token": 1.0
            }
        }

    def get(self, model_id: str):
        return self.models[model_id]["generator"]

    def cost_factor(self, model_id: str) -> float:
        return self.models[model_id]["cost_per_token"]
