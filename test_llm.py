
from llm import generate_answer

question = "What is Artificial Intelligence?"

passages = [
    {
        "text": "Artificial Intelligence is a technology that enables computers to perform tasks that normally require human intelligence."
    },
    {
        "text": "AI systems can learn from data, recognize patterns, and make decisions."
    }
]

answer = generate_answer(question, passages)

print("\nGenerated Answer:\n")
print(answer)
