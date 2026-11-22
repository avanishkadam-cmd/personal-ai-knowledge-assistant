
from vector_store import store_chunks, search_chunks

# Sample document chunks
chunks = [
    "Python is a popular programming language.",
    "Machine learning allows computers to learn from data.",
    "SQL is used to manage relational databases.",
    "Deep learning uses artificial neural networks."
]

# Store chunks in ChromaDB
count = store_chunks(chunks, "test_document")

print(f"\nStored {count} chunks successfully!")

# Ask a question
question = "How can computers learn automatically?"

# Search for relevant chunks
results = search_chunks(question, n_results=2)

print("\nSearch Results:")

for document in results["documents"][0]:
    print("-", document)
