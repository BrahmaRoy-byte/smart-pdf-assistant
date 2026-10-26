
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.

    chunk_size: maximum characters per chunk
    overlap: characters shared between consecutive chunks
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks



if __name__ == "__main__":
    sample_text = """
    Machine learning enables computers to learn from data.
    Linear regression is used to predict numerical values.
    Gradient descent helps optimize model parameters.
    Neural networks consist of interconnected computational units.
    """

    chunks = split_text_into_chunks(
        sample_text,
        chunk_size=100,
        overlap=20,
    )

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
