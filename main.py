from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai_helpers import embed_text, query_openai
from data_store import vector_store

app = FastAPI(title="RAG-Boost", description="Simple RAG pipeline", version="1.0")

class IngestRequest(BaseModel):
    id: str
    content: str

class QueryRequest(BaseModel):
    query: str

@app.post("/ingest")
def ingest(req: IngestRequest):
    embedding = embed_text(req.content)
    vector_store[req.id] = {
        "embedding": embedding,
        "content": req.content
    }
    return {"message": f"Document '{req.id}' stored."}

@app.post("/query")
def query_rag(req: QueryRequest):
    if not vector_store:
        raise HTTPException(status_code=404, detail="No documents in store.")

    query_embedding = embed_text(req.query)

    from numpy import dot
    from numpy.linalg import norm

    def cosine(a, b): return dot(a, b) / (norm(a) * norm(b))

    scored = [
        (doc_id, cosine(query_embedding, doc["embedding"]), doc["content"])
        for doc_id, doc in vector_store.items()
    ]
    top_doc = max(scored, key=lambda x: x[1])

    response = query_openai(req.query, context=top_doc[2])
    return {
        "most_similar_doc": top_doc[0],
        "similarity_score": top_doc[1],
        "gpt_response": response
    }
