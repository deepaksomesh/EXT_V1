from models.generator import Generator

class ModelRegistry:
    """
    Registry for multiple LLMs with cost coefficients.
    Supports both causal LM and seq2seq models.
    """

    def __init__(self):
        self.models = {
            "small": {
                "generator": Generator(model_name="distilgpt2", is_seq2seq=False),
                "cost_per_token": 0.5
            },
            "medium": {
                "generator": Generator(model_name="google/flan-t5-base", is_seq2seq=True),
                "cost_per_token": 1.5
            },
            "large": {
                "generator": Generator(model_name="google/flan-t5-large", is_seq2seq=True),
                "cost_per_token": 3.0
            }
        }

    def get(self, model_id: str):
        return self.models[model_id]["generator"]

    def cost_factor(self, model_id: str) -> float:
        return self.models[model_id]["cost_per_token"]

    def available_models(self):
        return list(self.models.keys())
