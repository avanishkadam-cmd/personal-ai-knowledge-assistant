
import chromadb
import numpy as np
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
DISTANCE_THRESHOLD = 0.9

# MMR relevance-diversity balan ce
MMR_LAMBDA = 0.9

# Retrieve a larger pool before applying MMR
CANDIDATE_MULTIPLIER = 3


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


def cosine_similarity(vector_a, vector_b):

    vector_a = np.asarray(vector_a)
    vector_b = np.asarray(vector_b)

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b) / denominator
    )


def mmr_select(
    question_embedding,
    candidate_embeddings,
    n_results,
    lambda_mult=MMR_LAMBDA
):

    selected_indices = []
    remaining_indices = list(
        range(len(candidate_embeddings))
    )

    while remaining_indices and len(selected_indices) < n_results:

        best_index = None
        best_score = float("-inf")

        for candidate_index in remaining_indices:

            candidate_embedding = candidate_embeddings[
                candidate_index
            ]

            # Relevance to the user's question
            relevance = cosine_similarity(
                question_embedding,
                candidate_embedding
            )

            # Similarity to already selected chunks
            diversity_penalty = 0

            if selected_indices:

                diversity_penalty = max(
                    cosine_similarity(
                        candidate_embedding,
                        candidate_embeddings[selected_index]
                    )
                    for selected_index in selected_indices
                )

            # MMR formula
            mmr_score = (
                lambda_mult * relevance
                - (1 - lambda_mult) * diversity_penalty
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = candidate_index

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return selected_indices


def search_chunks(question, n_results=3):

    if n_results <= 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    question_embedding = model.encode(question).tolist()

    total_chunks = collection.count()

    if total_chunks == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    # Retrieve a larger candidate pool
    candidate_count = min(
        n_results * CANDIDATE_MULTIPLIER,
        total_chunks
    )

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "distances",
            "embeddings"
        ]
    )

    # Keep only candidates passing the distance threshold
    candidates = []

    for document, metadata, distance, embedding in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["embeddings"][0]
    ):

        print("\nDistance:", round(float(distance), 4))

        if distance <= DISTANCE_THRESHOLD:

            print("Decision: KEEP")

            candidates.append({
                "document": document,
                "metadata": metadata,
                "distance": float(distance),
                "embedding": embedding
            })

        else:
            print("Decision: REJECT")

    if not candidates:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    # Apply MMR to select relevant and diverse chunks
    candidate_embeddings = [
        candidate["embedding"]
        for candidate in candidates
    ]

    selected_indices = mmr_select(
        question_embedding,
        candidate_embeddings,
        min(n_results, len(candidates))
    )

    selected_candidates = [
        candidates[index]
        for index in selected_indices
    ]

    print("\n--- MMR Selection ---")

    for candidate in selected_candidates:
        print(
            "Selected:",
            candidate["metadata"].get("filename"),
            "| Page:",
            candidate["metadata"].get("page"),
            "| Distance:",
            round(candidate["distance"], 4)
        )

    return {
        "documents": [[
            candidate["document"]
            for candidate in selected_candidates
        ]],
        "metadatas": [[
            candidate["metadata"]
            for candidate in selected_candidates
        ]],
        "distances": [[
            candidate["distance"]
            for candidate in selected_candidates
        ]]
    }
