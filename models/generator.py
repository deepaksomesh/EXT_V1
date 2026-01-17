from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# MODEL_NAME = "distilgpt2"

class Generator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, query, contexts):
        prompt = (
            "You are a helpful assistant.\n\n"
            "Use the following context to answer the question.\n\n"
        )

        if contexts:
            prompt += "Context:\n"
            for ctx in contexts:
                prompt += f"- {ctx}\n"
        else:
            prompt += "Context: None\n"

        prompt += f"\nQuestion: {query}\nAnswer:"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True
        )

        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False
            )
        
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        total_tokens = output.shape[1]
        completion_tokens = total_tokens - prompt_tokens

        return decoded, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
