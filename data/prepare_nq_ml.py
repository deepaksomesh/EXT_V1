from datasets import load_dataset
import json
from tqdm import tqdm

OUTPUT_DOCS = "data/processed/documents.json"
OUTPUT_QA = "data/processed/qa_pairs.json"

KEYWORDS = [
    "machine learning", "artificial intelligence", "neural",
    "algorithm", "data", "optimization", "gradient",
    "regression", "classification", "model", "computer science"
]

def is_ml_related(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in KEYWORDS)


def main(max_samples=3000):
    dataset = load_dataset(
        "natural_questions",
        "default",
        split="train",
        streaming=True
    )

    documents = []
    qa_pairs = []
    doc_id = 0

    for item in tqdm(dataset):
        annotations = item.get("annotations")
        if not annotations or not isinstance(annotations, dict):
            continue

        # ---- CORRECT UNWRAPPING (dict -> list -> dict) ----
        ann_list = next(iter(annotations.values()))
        if not ann_list or not isinstance(ann_list, list):
            continue

        ann = ann_list[0]
        if not isinstance(ann, dict):
            continue

        long_answer = ann.get("long_answer", {})
        start = long_answer.get("start_token", -1)
        end = long_answer.get("end_token", -1)

        if start == -1 or end <= start:
            continue

        tokens = item["document"]["tokens"]["token"]
        if end > len(tokens):
            continue

        paragraph = " ".join(tokens[start:end])
        if not is_ml_related(paragraph):
            continue

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

        documents.append({
            "doc_id": f"doc_{doc_id}",
            "text": paragraph
        })

        qa_pairs.append({
            "question": question,
            "answer": answer_text
        })

        doc_id += 1
        if doc_id >= max_samples:
            break

    with open(OUTPUT_DOCS, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    with open(OUTPUT_QA, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Saved {len(documents)} documents and QA pairs")


if __name__ == "__main__":
    main()
