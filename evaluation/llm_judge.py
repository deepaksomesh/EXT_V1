"""
LLM-as-Judge Evaluation Module

Uses a local LLM (Flan-T5) to evaluate answer quality by comparing
generated answers against gold answers on multiple dimensions:
- Correctness: Is the answer factually correct?
- Relevance: Does the answer address the question?
- Completeness: Is the answer complete?

Returns a 1-5 score that can be averaged across samples.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dataclasses import dataclass
from typing import List, Tuple
import re


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    score: int  # 1-5 scale
    reasoning: str
    
    def to_dict(self):
        return {"score": self.score, "reasoning": self.reasoning}


class LLMJudge:
    """
    Uses Flan-T5 as a judge to evaluate answer quality.
    
    Score scale:
    1 - Completely wrong or irrelevant
    2 - Partially relevant but mostly incorrect
    3 - Somewhat correct but incomplete
    4 - Mostly correct with minor issues
    5 - Fully correct and complete
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
    
    def _build_judge_prompt(self, question: str, gold_answer: str, generated_answer: str) -> str:
        """Build evaluation prompt for the LLM judge."""
        return f"""You are an expert evaluator. Rate the generated answer compared to the correct answer.

Question: {question}

Correct Answer: {gold_answer}

Generated Answer: {generated_answer}

Rate the generated answer on a scale of 1-5:
1 = Completely wrong
2 = Partially correct
3 = Somewhat correct but incomplete
4 = Mostly correct
5 = Fully correct

Score (just the number):"""
    
    def _parse_score(self, response: str) -> int:
        """Extract numeric score from LLM response."""
        # Try to find a number 1-5
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            return int(numbers[0])
        # Default to middle score if parsing fails
        return 3
    
    def judge(self, question: str, gold_answer: str, generated_answer: str) -> JudgeResult:
        """
        Evaluate a single answer using LLM-as-judge.
        
        Returns a JudgeResult with score (1-5) and reasoning.
        """
        if not generated_answer or not generated_answer.strip():
            return JudgeResult(score=1, reasoning="Empty answer")
        
        prompt = self._build_judge_prompt(question, gold_answer, generated_answer)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        score = self._parse_score(response)
        
        return JudgeResult(
            score=score,
            reasoning=f"LLM evaluation: {response.strip()}"
        )
    
    def judge_batch(self, samples: List[Tuple[str, str, str]]) -> dict:
        """
        Evaluate a batch of (question, gold_answer, generated_answer) tuples.
        
        Returns aggregate statistics.
        """
        scores = []
        results = []
        
        for question, gold, generated in samples:
            result = self.judge(question, gold, generated)
            scores.append(result.score)
            results.append(result)
        
        n = len(scores)
        if n == 0:
            return {}
        
        # Calculate distribution
        score_dist = {i: scores.count(i) / n for i in range(1, 6)}
        
        return {
            "mean_score": round(sum(scores) / n, 3),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_distribution": score_dist,
            "num_samples": n,
            "results": results
        }


class ComparativeJudge(LLMJudge):
    """
    Extended judge that directly compares two answers.
    Returns which answer is better: A, B, or Tie.
    """
    
    def _build_comparison_prompt(self, question: str, gold: str, answer_a: str, answer_b: str) -> str:
        return f"""Compare two answers to the same question.

Question: {question}
Correct Answer: {gold}

Answer A: {answer_a}
Answer B: {answer_b}

Which answer is better? Reply with only: A, B, or Tie"""
    
    def compare(self, question: str, gold: str, answer_a: str, answer_b: str) -> str:
        """Compare two answers and return winner: 'A', 'B', or 'Tie'."""
        prompt = self._build_comparison_prompt(question, gold, answer_a, answer_b)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True).strip().upper()
        
        if 'A' in response and 'B' not in response:
            return 'A'
        elif 'B' in response and 'A' not in response:
            return 'B'
        else:
            return 'Tie'
    
    def compare_batch(self, samples: List[Tuple[str, str, str, str]]) -> dict:
        """
        Compare batches of (question, gold, answer_a, answer_b).
        
        Returns win rates for A, B, and Tie.
        """
        results = {'A': 0, 'B': 0, 'Tie': 0}
        
        for question, gold, answer_a, answer_b in samples:
            winner = self.compare(question, gold, answer_a, answer_b)
            results[winner] += 1
        
        n = sum(results.values())
        if n == 0:
            return {}
        
        return {
            "A_wins": results['A'],
            "B_wins": results['B'],
            "ties": results['Tie'],
            "A_win_rate": round(results['A'] / n, 3),
            "B_win_rate": round(results['B'] / n, 3),
            "tie_rate": round(results['Tie'] / n, 3),
            "num_comparisons": n
        }
