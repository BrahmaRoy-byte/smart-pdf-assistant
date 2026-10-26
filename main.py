
from pdf_reader import extract_text_from_pdf
from text_chunker import split_text_into_chunks
from semantic_search import create_embeddings, search_chunks
from answer_generator import generate_answer

def process_pdf(pdf_path):
    print("Reading PDF...")

    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        print("No selectable text found in this PDF.")
        return

    print("Splitting text into chunks...")

    chunks = split_text_into_chunks(
        text,
        chunk_size=500,
        overlap=50,
    )

    if not chunks:
        print("No text chunks were created.")
        return

    print(f"Total chunks created: {len(chunks)}")

    print("\nCreating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Embeddings created successfully!")
    print("\nYou can now ask questions about your PDF.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        results = search_chunks(
            question,
            chunks,
            embeddings,
            top_k=3,
        )

        
        context = "\n\n".join(
            result["text"] for result in results
        )

        print("\nGenerating AI answer...")

        try:
            answer = generate_answer(question, context)

            print("\nAI Answer:")
            print(answer)

        except Exception as error:
            print(f"Could not generate answer: {error}")

        print("\nRelevant passages:")

        for index, result in enumerate(results, start=1):
            print(f"\n--- Result {index} ---")
            print(f"Similarity: {result['score']:.3f}")
            print(result["text"])


if __name__ == "__main__":
    pdf_path = input("Enter your PDF path: ").strip().strip('"')
    process_pdf(pdf_path)
