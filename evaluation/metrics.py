"""
Enhanced evaluation metrics for RAG system.
Includes: Semantic Similarity, Exact Match, Token-level Precision/Recall/F1
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Comprehensive evaluation result for a single prediction."""
    semantic_similarity: float
    exact_match: float
    token_precision: float
    token_recall: float
    token_f1: float

    def to_dict(self) -> Dict:
        return {
            "semantic_similarity": round(self.semantic_similarity, 4),
            "exact_match": self.exact_match,
            "token_precision": round(self.token_precision, 4),
            "token_recall": round(self.token_recall, 4),
            "token_f1": round(self.token_f1, 4)
        }


class EnhancedMetrics:
    """
    Comprehensive evaluation metrics for question answering.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison: lowercase, remove punctuation, strip whitespace."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def tokenize(self, text: str) -> set:
        """Convert text to set of normalized tokens."""
        normalized = self.normalize_text(text)
        return set(normalized.split()) if normalized else set()

    def semantic_similarity(self, generated: str, gold: str) -> float:
        """Compute cosine similarity between sentence embeddings."""
        if not generated or not gold:
            return 0.0
        emb = self.encoder.encode([generated, gold])
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        return float(max(0.0, sim))

    def exact_match(self, generated: str, gold: str) -> float:
        """Check if normalized texts match exactly."""
        return 1.0 if self.normalize_text(generated) == self.normalize_text(gold) else 0.0

    def token_precision_recall_f1(self, generated: str, gold: str) -> tuple:
        """
        Compute token-level precision, recall, and F1.
        
        Precision = |generated ∩ gold| / |generated|
        Recall = |generated ∩ gold| / |gold|
        F1 = 2 * (precision * recall) / (precision + recall)
        """
        gen_tokens = self.tokenize(generated)
        gold_tokens = self.tokenize(gold)

        if not gen_tokens and not gold_tokens:
            return 1.0, 1.0, 1.0  # Both empty = perfect match

        if not gen_tokens:
            return 0.0, 0.0, 0.0

        if not gold_tokens:
            return 0.0, 0.0, 0.0

        overlap = gen_tokens & gold_tokens
        precision = len(overlap) / len(gen_tokens)
        recall = len(overlap) / len(gold_tokens)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    def evaluate(self, generated: str, gold: str) -> EvalResult:
        """Compute all metrics for a single prediction."""
        sem_sim = self.semantic_similarity(generated, gold)
        em = self.exact_match(generated, gold)
        precision, recall, f1 = self.token_precision_recall_f1(generated, gold)

        return EvalResult(
            semantic_similarity=sem_sim,
            exact_match=em,
            token_precision=precision,
            token_recall=recall,
            token_f1=f1
        )

    def evaluate_batch(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Evaluate a batch of predictions and return aggregate metrics.
        """
        results = []
        for pred, ref in zip(predictions, references):
            results.append(self.evaluate(pred, ref))

        n = len(results)
        if n == 0:
            return {}

        return {
            "mean_semantic_similarity": round(sum(r.semantic_similarity for r in results) / n, 4),
            "mean_exact_match": round(sum(r.exact_match for r in results) / n, 4),
            "mean_token_precision": round(sum(r.token_precision for r in results) / n, 4),
            "mean_token_recall": round(sum(r.token_recall for r in results) / n, 4),
            "mean_token_f1": round(sum(r.token_f1 for r in results) / n, 4),
            "num_samples": n
        }
