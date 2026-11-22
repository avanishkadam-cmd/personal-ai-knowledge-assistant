
import pymupdf
import re


def extract_text_from_pdf(pdf_bytes: bytes) -> dict:

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    extracted_pages = []

    try:
        for page_number, page in enumerate(document):

            text = clean_text(page.get_text("text"))

            extracted_pages.append({
                "page": page_number + 1,
                "text": text
            })

    finally:
        document.close()

    return {
        "total_pages": len(extracted_pages),
        "pages": extracted_pages
    }


def clean_text(text: str):

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def create_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    lines = text.splitlines()

    sections = []
    current_section = []

    heading_pattern = r'^\s*\d+\.\s+.+\?\s*$'

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Check whether this line is a numbered heading
        if re.match(heading_pattern, line):

            if current_section:
                sections.append(" ".join(current_section))

            current_section = [line]

        else:
            current_section.append(line)

    if current_section:
        sections.append(" ".join(current_section))

    # Split sections only if they exceed chunk_size
    chunks = []

    for section in sections:

        if len(section) <= chunk_size:
            chunks.append(section)

        else:
            sentences = re.split(
                r'(?<=[.!?])\s+',
                section
            )

            current_chunk = []

            for sentence in sentences:

                candidate = " ".join(
                    current_chunk + [sentence]
                )

                if len(candidate) <= chunk_size:
                    current_chunk.append(sentence)

                else:

                    if current_chunk:
                        chunks.append(
                            " ".join(current_chunk)
                        )

                    current_chunk = [sentence]

            if current_chunk:
                chunks.append(
                    " ".join(current_chunk)
                )

    return chunks


def create_page_chunks(pages: list[dict]) -> list[dict]:

    all_chunks = []

    for page in pages:

        chunks = create_chunks(page["text"])

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return all_chunks
