
from pdf_processor import extract_text_from_pdf, create_page_chunks

filename = "simple_ai_test_document.pdf"

with open(filename, "rb") as file:
    pdf_bytes = file.read()

result = extract_text_from_pdf(pdf_bytes)

chunks = create_page_chunks(result["pages"])

print("Total Pages:", result["total_pages"])
print("Total Chunks:", len(chunks))

for i, chunk in enumerate(chunks):

    print(f"\nCHUNK {i + 1}")
    print("Page:", chunk["page"])
    print("Characters:", len(chunk["text"]))
    print(chunk["text"])
    print("-" * 50)
