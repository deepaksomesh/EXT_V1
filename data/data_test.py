from datasets import load_dataset
import json
from tqdm import tqdm

# ======================
# Configuration
# ======================
OUTPUT_DOCS = "data/processed/documents.json"
OUTPUT_QA = "data/processed/qa_pairs.json"

MAX_SAMPLES = 800        # final dataset size (documents + QA)
MAX_SCANNED = 8000       # hard cap on streamed examples

KEYWORDS = [
    "machine learning", "artificial intelligence", "neural",
    "algorithm", "data", "optimization", "gradient",
    "regression", "classification", "model", "computer science",
    "loss function", "training data", "inference"
]

# ======================
# Helpers
# ======================
def is_ml_related(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in KEYWORDS)

# ======================
# Main
# ======================
def main():
    dataset = load_dataset(
        "natural_questions",
        "default",
        split="train",
        streaming=True
    )

    documents = []
    qa_pairs = []

    doc_id = 0
    scanned = 0

    for item in tqdm(dataset):
        scanned += 1
        if scanned >= MAX_SCANNED:
            break

        # --------------------------
        # Annotations (STREAMING SAFE)
        # annotations: dict -> list -> dict
        # --------------------------
        annotations = item.get("annotations")
        if not annotations or not isinstance(annotations, dict):
            continue

        ann_list = next(iter(annotations.values()))
        if not ann_list or not isinstance(ann_list, list):
            continue

        ann = ann_list[0]
        if not isinstance(ann, dict):
            continue

        # --------------------------
        # Long answer (paragraph)
        # --------------------------
        long_answer = ann.get("long_answer", {})
        start = long_answer.get("start_token", -1)
        end = long_answer.get("end_token", -1)

        if start == -1 or end <= start:
            continue

        tokens = item["document"]["tokens"]["token"]
        if end > len(tokens):
            continue

        # Early filtering (first 200 tokens only)
        preview_text = " ".join(tokens[start : min(start + 200, end)])
        if not is_ml_related(preview_text):
            continue

        paragraph = " ".join(tokens[start:end])

        # --------------------------
        # Short answer (ground truth)
        # --------------------------
        short_answers = ann.get("short_answers", [])
        if not short_answers:
            continue

        short = short_answers[0]
        sa_start = short.get("start_token", -1)
        sa_end = short.get("end_token", -1)

        if sa_start == -1 or sa_end <= sa_start:
            continue

        answer_text = " ".join(tokens[sa_start:sa_end])

        question = item["question"]["text"]

        # --------------------------
        # Store
        # --------------------------
        documents.append({
            "doc_id": f"doc_{doc_id}",
            "text": paragraph
        })

        qa_pairs.append({
            "question": question,
            "answer": answer_text
        })

        doc_id += 1
        if doc_id >= MAX_SAMPLES:
            break

    # ======================
    # Save
    # ======================
    with open(OUTPUT_DOCS, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    with open(OUTPUT_QA, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"\n✅ Done")
    print(f"Scanned examples : {scanned}")
    print(f"Saved documents  : {len(documents)}")
    print(f"Saved QA pairs   : {len(qa_pairs)}")


if __name__ == "__main__":
    main()
