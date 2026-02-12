"""
Research-Grade Contextual Bandit Policy using LinUCB.

Algorithm: Disjoint LinUCB
Reference: Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation", WWW 2010.

Key Features:
- Maintains separate ridge regression models (A, b) for each arm.
- Selects actions based on Upper Confidence Bound (UCB).
- Balances exploration (uncertainty) and exploitation (reward prediction).
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import random
from typing import Dict, List, Optional

class LinUCBPolicy:
    """
    Disjoint LinUCB for contextual bandits.
    """

    ACTIONS = [
        {"k": 3, "model": "small"},   # Low cost, low latency
        {"k": 5, "model": "small"},   # Medium-low cost
        {"k": 3, "model": "medium"},  # Medium-high cost
        {"k": 5, "model": "medium"},  # High cost, best quality
    ]

    def __init__(self, alpha: float = 1.0, feature_dim: int = 11):
        """
        Args:
            alpha: Exploration parameter (higher = more exploration).
            feature_dim: Dimension of context vector (1 token count + 10 embedding dims).
        """
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize A (covariance) and b (reward) for each arm
        # A is d*d identity matrix, b is d*1 zero vector
        self.A = [np.identity(feature_dim) for _ in self.ACTIONS]
        self.b = [np.zeros((feature_dim, 1)) for _ in self.ACTIONS]
        
        # Function to compute inverse of A (lazy cache could be added)
        self.A_inv = [np.identity(feature_dim) for _ in self.ACTIONS]

    def featurize(self, query: str) -> np.ndarray:
        """Convert query to feature vector."""
        tokens = len(query.split())
        # Normalized token count to avoid huge values
        norm_tokens = min(tokens / 50.0, 1.0) 
        
        emb = self.embedder.encode(query)
        # Use first 10 dims of embedding for efficiency + bias term
        # (Actually, better to use PCA, but slicing is a standard "cheap" projection)
        context = np.concatenate([[norm_tokens], emb[:10]])
        return context.reshape(-1, 1) # (d, 1)

    def select_action(self, query: str) -> Dict:
        """Select action using UCB."""
        x = self.featurize(query)
        p_t = np.zeros(len(self.ACTIONS))

        for a in range(len(self.ACTIONS)):
            # Ridge regression estimate: theta_hat = A_inv * b
            theta_hat = np.dot(self.A_inv[a], self.b[a])
            
            # Upper Confidence Bound calculation
            # UCB = mean + alpha * std_dev
            # std_dev = sqrt(x.T * A_inv * x)
            expected_reward = np.dot(theta_hat.T, x).item()
            uncertainty = self.alpha * np.sqrt(np.dot(np.dot(x.T, self.A_inv[a]), x).item())
            
            p_t[a] = expected_reward + uncertainty

        # Break ties randomly
        best_action_idx = np.random.choice(np.flatnonzero(p_t == p_t.max()))
        return self.ACTIONS[best_action_idx]

    def train(self, query: str, action_idx: int, reward: float):
        """Update bandit parameters with observed reward."""
        x = self.featurize(query)
        
        # Update A and b for the chosen arm
        self.A[action_idx] += np.dot(x, x.T)
        self.b[action_idx] += reward * x
        
        # Recompute inverse (Sherman-Morrison could optimize this to O(d^2) instead of O(d^3))
        # For d=11, np.linalg.inv is fast enough
        self.A_inv[action_idx] = np.linalg.inv(self.A[action_idx])

    def get_diagnostics(self):
        """Return diagnostic info about learner state."""
        return {
            "alpha": self.alpha,
            "arm_confidence_norms": [np.linalg.norm(A_inv) for A_inv in self.A_inv],
            "estimated_thetas": [
                np.dot(self.A_inv[a], self.b[a]).flatten().tolist() 
                for a in range(len(self.ACTIONS))
            ]
        }
