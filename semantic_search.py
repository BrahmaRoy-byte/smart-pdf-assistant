
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pretrained text embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def create_embeddings(chunks):
    """Convert text chunks into numerical vectors."""

    texts = [chunk["text"] if isinstance(chunk, dict) else chunk
             for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings


def search_chunks(question, chunks, embeddings, top_k=5):
    """Find relevant chunks, optionally limited to a named PDF."""

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    # Check whether the question mentions a PDF filename
    question_lower = question.lower()

    matching_indices = [
        i for i, chunk in enumerate(chunks)
        if isinstance(chunk, dict)
        and chunk.get("source", "").lower() in question_lower
        and chunk.get("source", "").lower().endswith(".pdf")
    ]

    # If a PDF filename is mentioned, search only that PDF
    if matching_indices:
        candidate_indices = matching_indices
    else:
        candidate_indices = list(range(len(chunks)))

    scores = embeddings @ question_embedding

    # Rank only the eligible chunks by similarity
    best_indices = sorted(
        candidate_indices,
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for index in best_indices:
        chunk = chunks[index]
        text = chunk["text"] if isinstance(chunk, dict) else chunk

        results.append({
            "text": text,
            "score": float(scores[index]),
            "source": chunk.get("source", "Unknown")
                      if isinstance(chunk, dict)
                      else "Unknown"
        })

    return results
