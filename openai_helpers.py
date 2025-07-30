import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_text(text: str) -> list:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def query_openai(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You answer questions based on provided context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response["choices"][0]["message"]["content"]
