import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "data/index.faiss"
DOC_PATH = "data/processed/documents.json"

def main():
    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    texts = [d["text"] for d in docs]

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)

    print(f"Indexed {len(texts)} documents")

if __name__ == "__main__":
    main()
