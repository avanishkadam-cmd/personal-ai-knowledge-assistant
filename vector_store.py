
import chromadb
from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Initialize persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")


# Create or access collection
collection = client.get_or_create_collection(
    name="knowledge_base"
)


# Relevance threshold (experimental)
DISTANCE_THRESHOLD = 1.2


def store_chunks(chunks, filename):

    if not chunks:
        return 0

    # Remove old chunks belonging to this file
    collection.delete(
        where={"filename": filename}
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts).tolist()

    ids = [
        f"{filename}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "filename": filename,
            "page": int(chunk["page"]),
            "chunk": i
        }
        for i, chunk in enumerate(chunks)
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def search_chunks(question, n_results=3):

    question_embedding = model.encode(question).tolist()

    total_chunks = collection.count()

    if total_chunks == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    # Retrieve Top-K relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(n_results, total_chunks)
    )

    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        print("\nDistance:", round(distance, 4))

        if distance <= DISTANCE_THRESHOLD:

            print("Decision: KEEP")

            filtered_documents.append(document)
            filtered_metadatas.append(metadata)
            filtered_distances.append(distance)

        else:
            print("Decision: REJECT")

    return {
        "documents": [filtered_documents],
        "metadatas": [filtered_metadatas],
        "distances": [filtered_distances]
    }
