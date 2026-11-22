
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import traceback

from pdf_processor import extract_text_from_pdf, create_page_chunks
from vector_store import store_chunks, search_chunks
from llm import generate_answer


app = FastAPI(
    title="AI Knowledge Assistant",
    description="Ask questions about your documents",
    version="1.0.0"
)


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Welcome to AI Knowledge Assistant!",
        "status": "Running"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask")
def ask_question(data: Question):

    if not data.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        # Step 1: Search relevant PDF chunks
        results = search_chunks(data.question, n_results=3)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Step 2: Prepare retrieved passages
        sources = []

        for document, metadata in zip(documents, metadatas):
            sources.append({
                "text": document,
                "filename": metadata.get("filename", "Unknown"),
                "page": metadata.get("page", "Unknown")
            })

        # Step 3: Generate an answer using Llama
        if sources:
            answer = generate_answer(
                data.question,
                sources
            )
        else:
            answer = "No relevant information was found in your uploaded documents."

        # Step 4: Return answer and sources
        return {
            "question": data.question,
            "answer": answer,
            "total_sources": len(sources),
            "sources": [
                {
                    "filename": source["filename"],
                    "page": source["page"],
                    "text": source["text"]
                }
                for source in sources
            ]
        }

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {type(e).__name__}: {str(e)}"
        )


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file"
        )

    try:
        pdf_bytes = await file.read()

        if not pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty"
            )

        result = extract_text_from_pdf(pdf_bytes)

        chunks = create_page_chunks(result["pages"])

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in this PDF"
            )

        stored_count = store_chunks(
            chunks,
            file.filename
        )

        return {
            "filename": file.filename,
            "total_pages": result["total_pages"],
            "total_chunks": len(chunks),
            "stored_chunks": stored_count,
            "status": "PDF successfully processed and stored"
        }

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {type(e).__name__}: {str(e)}"
        )

    finally:
        await file.close()
