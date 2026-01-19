import wikipedia
import json
import re

OUTPUT_DOCS = "data/processed/documents.json"
OUTPUT_QA = "data/processed/qa_pairs.json"

# --------------------------
# Fixed list of ML / CS pages
# --------------------------
WIKI_PAGES = [
    "Machine learning",
    "Artificial intelligence",
    "Deep learning",
    "Neural network",
    "Supervised learning",
    "Unsupervised learning",
    "Reinforcement learning",
    "Gradient descent",
    "Backpropagation",
    "Support vector machine",
    "Decision tree learning",
    "Random forest",
    "Linear regression",
    "Logistic regression",
    "K-means clustering",
    "Principal component analysis",
    "Bias–variance tradeoff",
    "Overfitting",
    "Underfitting",
    "Loss function",
    "Optimization algorithm",
    "Feature engineering",
    "Model evaluation",
    "Cross-validation",
    "Artificial neural network",
    "Convolutional neural network",
    "Recurrent neural network",
    "Transformer (machine learning)",
    "Attention mechanism"
]

def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    return text.strip()

def split_into_paragraphs(text, min_len=300):
    paras = []
    for p in text.split("\n"):
        p = clean_text(p)
        if len(p) >= min_len:
            paras.append(p)
    return paras

def generate_qa_from_paragraph(paragraph, title):
    """
    Simple but defensible QA generation.
    """
    question = f"What is {title}?"
    answer = paragraph.split(".")[0] + "."
    return question, answer

def main():
    documents = []
    qa_pairs = []
    doc_id = 0

    for title in WIKI_PAGES:
        try:
            page = wikipedia.page(title, auto_suggest=False)
        except Exception as e:
            print(f"Skipping {title}: {e}")
            continue

        paragraphs = split_into_paragraphs(page.content)

        for p in paragraphs:
            documents.append({
                "doc_id": f"doc_{doc_id}",
                "text": p
            })

            q, a = generate_qa_from_paragraph(p, title)
            qa_pairs.append({
                "question": q,
                "answer": a
            })

            doc_id += 1

    with open(OUTPUT_DOCS, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    with open(OUTPUT_QA, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)

    print("✅ Wikipedia ML dataset created")
    print(f"Documents: {len(documents)}")
    print(f"QA pairs : {len(qa_pairs)}")

if __name__ == "__main__":
    main()
