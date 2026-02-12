# Adaptive GraphRAG: Intelligent Retrieval-Augmented Generation with Contextual Bandits

A research-oriented RAG system that goes beyond static retrieve-then-generate pipelines. Instead of using fixed retrieval depths and a single language model for every query, this system **learns** the best retrieval and generation strategy per query using contextual bandits — specifically, the **LinUCB** algorithm.

It also incorporates **GraphRAG** — a knowledge graph built from document corpora using LLM-based triplet extraction — and fuses graph-derived context with traditional dense (FAISS) retrieval to produce richer, more grounded answers.

---

## Why This Project Exists

Most RAG systems treat every query the same: retrieve 5 documents, feed them to one model, return the answer. But not all questions are equal:

- **"What is gradient descent?"** — A simple factual lookup. A small, fast model with a few retrieved passages is more than enough.
- **"Explain the bias-variance tradeoff in the context of neural network regularization"** — A complex, nuanced question that benefits from more retrieved context and a larger, more capable model.

Wasting a large model on simple queries costs tokens (and time). Under-serving complex queries with a small model produces poor answers. This project tackles that problem head-on.

**The core idea:** Use a contextual bandit (LinUCB) to jointly decide, per query:
1. **How many documents to retrieve** (`k = 3` or `k = 5`)
2. **Which language model to use** (`small` = DistilGPT2, `medium` = Flan-T5-Base)

The bandit learns from a reward signal that balances answer quality, cost, and latency — continuously improving its routing decisions over time.

---

## Architecture Overview

```
                              ┌─────────────────────┐
                              │    User Query        │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   LinUCB Policy      │
                              │  (select k, model)   │
                              └──────────┬──────────┘
                                         │
                          ┌──────────────┼──────────────┐
                          │                             │
               ┌──────────▼──────────┐      ┌──────────▼──────────┐
               │   Dense Retrieval   │      │   Graph Retrieval   │
               │      (FAISS)        │      │    (Knowledge Graph) │
               └──────────┬──────────┘      └──────────┬──────────┘
                          │                             │
                          └──────────┬──────────────────┘
                                     │ Hybrid Fusion
                          ┌──────────▼──────────┐
                          │   LLM Generation     │
                          │  (selected model)    │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Answer + Metrics   │
                          │  (logged for learning)│
                          └──────────────────────┘
```

---

## Key Components

### 1. Data Pipeline (`data/`)

Two data sourcing strategies are included:

- **Wikipedia Ingestion** (`wiki.py`) — Scrapes 28 Wikipedia pages on ML/CS topics (machine learning, neural networks, SVMs, transformers, etc.), chunks them into paragraphs, and auto-generates QA pairs from the first sentence of each paragraph.
- **Natural Questions** (`prepare_nq_ml.py`) — Filters Google's Natural Questions dataset for ML-related content, extracting question-answer pairs along with their source paragraphs.

Both scripts output:
- `data/processed/documents.json` — The document corpus used for retrieval
- `data/processed/qa_pairs.json` — QA pairs used for evaluation and policy training

### 2. Indexing (`indexing/`)

- **Dense Index** (`build_index.py`) — Encodes all documents using `all-MiniLM-L6-v2` (a sentence-transformer) and stores the embeddings in a FAISS flat L2 index for fast nearest-neighbor search.
- **Knowledge Graph** (`graph_builder.py`) — Uses Flan-T5-Large with one-shot prompting to extract `(Subject, Relation, Object)` triplets from documents. The resulting graph is stored as a GML file using NetworkX.

### 3. Retrieval (`retrieval/`)

The retrieval layer supports two modes that can be combined:

- **Dense Retrieval** (`retriever.py`) — Standard FAISS-based semantic search. Encodes the query, searches the index, and returns the top-k documents.
- **Graph Retrieval** (`graph_retriever.py`) — Performs entity linking (string matching) against the knowledge graph, then expands matched nodes by 1 hop to surface relational context (e.g., *"Apple Inc. founded by Steve Jobs"*). Graph results are appended to dense results with a `[Graph]` prefix.

When both are enabled, the system performs **hybrid retrieval** — combining vector similarity with structured knowledge.

### 4. Generation (`models/`)

- **Unified Generator** (`generator.py`) — A single class that handles both causal LM (GPT-style) and seq2seq (T5-style) models. It builds appropriate prompts for each architecture and tracks token usage.
- **Model Registry** (`model_registry.py`) — Manages multiple models with associated cost factors:

  | Model ID | Backbone | Type | Cost Factor |
  |----------|----------|------|-------------|
  | `small` | DistilGPT2 | Causal LM | 0.5 |
  | `medium` | Flan-T5-Base | Seq2Seq | 1.5 |
  | `large` | Flan-T5-Large | Seq2Seq | 3.0 |

### 5. Adaptive Policy (`policy/`)

This is the heart of the system — the decision-making layer that learns which (retrieval depth, model) combination works best for each query.

- **LinUCB Policy** (`linucb_policy.py`) — Implements the Disjoint LinUCB algorithm ([Li et al., WWW 2010](https://arxiv.org/abs/1003.0146)). Each possible action `(k, model)` is an arm. The policy:
  - Encodes the query into an 11-dimensional feature vector (normalized token count + first 10 dims of the sentence embedding)
  - Maintains per-arm ridge regression estimates (`A`, `b` matrices)
  - Selects actions by Upper Confidence Bound: `UCB = θ̂ᵀx + α√(xᵀA⁻¹x)`
  - Updates parameters online as rewards arrive

  The action space consists of 4 arms:

  | Arm | Retrieval k | Model | Use Case |
  |-----|-------------|-------|----------|
  | 0 | 3 | small | Simple factual queries |
  | 1 | 5 | small | Broader factual queries |
  | 2 | 3 | medium | Focused complex queries |
  | 3 | 5 | medium | Deep, nuanced queries |

- **Joint Decision Policy** (`joint_policy.py`) — An alternative ε-greedy policy using logistic regression. Useful as a baseline or for transfer learning scenarios.

- **Data Logger** (`data_logger.py`) — Logs every query interaction (query, action, answer, cost, latency, tokens) to a JSONL file for offline analysis and policy training.

### 6. Evaluation (`evaluation/`)

A multi-layered evaluation framework designed to rigorously assess both answer quality and policy effectiveness:

- **Metrics** (`metrics.py`) — Computes semantic similarity (cosine on sentence embeddings), exact match, and token-level precision/recall/F1 between generated and gold answers.
- **LLM-as-Judge** (`llm_judge.py`) — Uses Flan-T5 as an automated evaluator that scores answers on a 1–5 scale for correctness, relevance, and completeness.
- **Comparative Judge** (`llm_judge.py`) — Extends the judge to perform head-to-head comparisons between two policies' answers, returning A wins / B wins / Tie.
- **Robust Comparison** (`robust_compare.py`) — Runs multiple evaluation rounds across 30–50 samples, aggregates LLM judge scores and head-to-head results, and computes cost savings.
- **Reward Function** (`reward.py`) — Defines the composite reward: `reward = similarity − λ_cost × cost − λ_latency × latency`, balancing quality against efficiency.
- **Offline Training** (`offline_train.py`) — Replays logged interactions to retrain the policy on high-reward experiences.

### 7. API (`api/`)

A **FastAPI** server (`app.py`) that exposes the full pipeline through a single endpoint:

```
POST /query
{
    "query": "What is backpropagation?",
    "use_dataset": false,
    "use_graph": true
}
```

**Response:**
```json
{
    "query": "What is backpropagation?",
    "action": {"k": 3, "model": "small"},
    "answer": "Backpropagation is an algorithm for...",
    "metrics": {
        "latency_sec": 0.42,
        "cost": 85.0,
        "total_tokens": 170
    }
}
```

Set `use_dataset: true` to sample from the QA dataset (useful for training/evaluation loops). Set `use_graph: true` to enable hybrid dense + graph retrieval.

### 8. Monitoring (`monitoring/`)

- **Timer** — Context manager for latency measurement
- **Cost Model** — Computes normalized cost based on token usage and per-model cost factors
- **Query Metrics** — Structured dataclass for prompt tokens, completion tokens, latency, and cost

---

## Getting Started

### Prerequisites

- Python 3.10+
- ~4 GB disk space for models (downloaded on first run)

### Installation

```bash
git clone https://github.com/your-username/adaptive-graphrag.git
cd adaptive-graphrag

pip install -r requirements.txt
```

### Prepare the Data

**Option A: Wikipedia (recommended for quick start)**
```bash
python -m data.wiki
```

**Option B: Natural Questions (larger, richer dataset)**
```bash
pip install datasets tqdm
python -m data.prepare_nq_ml
```

### Build the Index

```bash
# Dense FAISS index
python -m indexing.build_index

# Knowledge Graph (uses Flan-T5-Large, ~3 GB download)
python -m indexing.graph_builder
```

### Run the API

```bash
uvicorn api.app:app --reload
```

Then visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Run Evaluation

```bash
# Quick evaluation
python -m evaluation.simple_eval

# Robust comparison (Fixed vs Adaptive policy with LLM-as-Judge)
python -m evaluation.robust_compare

# Offline policy training from logged data
python -m evaluation.offline_train
```

---

## Project Structure

```
├── api/
│   └── app.py                  # FastAPI server with full RAG pipeline
├── data/
│   ├── wiki.py                 # Wikipedia ML dataset builder
│   ├── prepare_nq_ml.py        # Natural Questions dataset loader
│   ├── qa_loader.py            # QA pair sampler for training/eval
│   └── processed/              # Generated documents and QA pairs
├── indexing/
│   ├── build_index.py          # FAISS index builder
│   └── graph_builder.py        # Knowledge graph constructor (LLM-based)
├── retrieval/
│   ├── retriever.py            # Hybrid retriever (Dense + Graph)
│   └── graph_retriever.py      # Knowledge graph traversal
├── models/
│   ├── generator.py            # Unified LLM generator (causal + seq2seq)
│   └── model_registry.py       # Multi-model registry with cost tracking
├── policy/
│   ├── linucb_policy.py        # LinUCB contextual bandit (primary)
│   ├── joint_policy.py         # ε-greedy joint policy (baseline)
│   ├── model_policy.py         # Standalone model selection policy
│   └── data_logger.py          # Interaction logger for offline learning
├── evaluation/
│   ├── metrics.py              # Semantic similarity, EM, token F1
│   ├── llm_judge.py            # LLM-as-Judge + Comparative Judge
│   ├── robust_compare.py       # Multi-run policy comparison framework
│   ├── reward.py               # Composite reward function
│   ├── quality.py              # Semantic similarity scorer
│   ├── offline_train.py        # Offline policy training from logs
│   └── simple_eval.py          # Quick evaluation script
├── monitoring/
│   └── metric.py               # Timer, cost model, query metrics
├── logs/
│   └── bandit_data.jsonl       # Logged interactions for offline learning
├── requirements.txt
└── README.md
```

---

## How the Learning Loop Works

```
  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
  │  Query       │────▶│ LinUCB picks │────▶│ RAG generates   │
  │  arrives     │     │ (k, model)   │     │ answer          │
  └─────────────┘     └──────────────┘     └────────┬────────┘
                                                     │
                                           ┌─────────▼────────┐
                                           │ Compute reward    │
                                           │ (quality - cost   │
                                           │  - latency)       │
                                           └─────────┬────────┘
                                                     │
                                           ┌─────────▼────────┐
                                           │ Update bandit     │
                                           │ parameters        │
                                           │ (A, b matrices)   │
                                           └──────────────────┘
```

Over time, the bandit converges on routing simple queries to lightweight configurations and complex queries to more powerful ones — achieving better quality-cost tradeoffs than a one-size-fits-all approach.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense Retrieval | FAISS (Flat L2) |
| Graph Storage | NetworkX (MultiDiGraph, GML format) |
| Triplet Extraction | Google Flan-T5-Large |
| Generation (small) | DistilGPT2 |
| Generation (medium) | Google Flan-T5-Base |
| Generation (large) | Google Flan-T5-Large |
| Bandit Algorithm | Disjoint LinUCB |
| API Framework | FastAPI + Uvicorn |
| Evaluation | LLM-as-Judge (Flan-T5) + Token F1 + Semantic Similarity |

---

## References

- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation.* WWW 2010.
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv.

---

## License

This project is for research and educational purposes.
