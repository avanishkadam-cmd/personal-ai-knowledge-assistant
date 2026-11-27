str
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uuid
import traceback

from pdf_processor import extract_text_from_pdf, create_page_chunks
from vector_store import store_chunks, search_chunks, list_documents
from llm import generate_answer
from vector_store import (
    store_chunks,
    search_chunks,
    list_documents,
    delete_document
)

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

@app.delete("/documents/{document_id}")
def remove_document(document_id: str):

    try:
        deleted_count = delete_document(document_id)

        if deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )

        return {
            "document_id": document_id,
            "deleted_chunks": deleted_count,
            "status": "Document successfully deleted"
        }

    except HTTPException:
        raise

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Could not delete document: {str(e)}"
        )

@app.get("/documents")
def get_documents():

    try:
        documents = list_documents()

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve documents: {str(e)}"
        )
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

        for i, (document, metadata) in enumerate(
            zip(documents, metadatas),
            start=1
        ):
            sources.append({
                "source_id": i,
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
                    "source_id": source["source_id"],
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

        document_id = str(uuid.uuid4())

        stored_count = store_chunks(
            chunks,
            file.filename,
            document_id
        )

        return {
            "filename": file.filename,
            "total_pages": result["total_pages"],
            "total_chunks": len(chunks),
            "stored_chunks": stored_count,
            "status": "PDF successfully processed and stored",
            "document_id": document_id,
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
@app.get("/debug-documents")
def debug_documents_endpoint():

    from vector_store import debug_documents

    debug_documents()

    return {"message": "Check your terminal"}