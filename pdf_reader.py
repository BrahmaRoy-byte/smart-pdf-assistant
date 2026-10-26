
import pymupdf


def extract_text_from_pdf(pdf_path):
    with pymupdf.open(pdf_path) as document:
        full_text = ""

        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text")

            full_text += f"\n--- Page {page_number} ---\n"
            full_text += page_text

    return full_text


if __name__ == "__main__":
    pdf_path = input("Enter the path to your PDF: ")

    try:
        text = extract_text_from_pdf(pdf_path)

        if text.strip():
            print("\nPDF text extracted successfully!\n")
            print(text)
        else:
            print("No selectable text found in this PDF.")

    except Exception as error:
        print(f"Error reading PDF: {error}")
