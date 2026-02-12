from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

class Generator:
    """
    Unified generator supporting both causal LM (GPT-style) and seq2seq (T5-style) models.
    """

    def __init__(self, model_name: str, is_seq2seq: bool = False):
        self.model_name = model_name
        self.is_seq2seq = is_seq2seq
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.eval()

    def generate(self, query, contexts):
        if self.is_seq2seq:
            prompt = self._build_seq2seq_prompt(query, contexts)
        else:
            prompt = self._build_causal_prompt(query, contexts)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )

        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if self.is_seq2seq:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
                # For seq2seq, output doesn't include input
                decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                total_tokens = prompt_tokens + output.shape[1]
                completion_tokens = output.shape[1]
            else:
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

    def _build_seq2seq_prompt(self, query, contexts):
        """Build prompt optimized for instruction-tuned seq2seq models like Flan-T5."""
        prompt = "Answer the question based on the context.\n\n"

        if contexts:
            prompt += "Context:\n"
            for ctx in contexts:
                prompt += f"- {ctx}\n"
        else:
            prompt += "Context: None\n"

        prompt += f"\nQuestion: {query}\nAnswer:"
        return prompt

    def _build_causal_prompt(self, query, contexts):
        """Build prompt for causal LM models like GPT-2."""
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
        return prompt
