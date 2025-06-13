from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import httpx
import os

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(host="localhost", port=6333)
collection_name = "tds_posts"

class Query(BaseModel):
    question: str

OLLAMA_URL = "http://localhost:11434"  # Ollama API URL

@app.post("/api/")
async def answer_question(query: Query):
    q_embedding = model.encode(query.question).tolist()

    # Search Qdrant
    hits = qdrant.search(collection_name, query_vector=q_embedding, limit=3)
    context_texts = []
    links = []
    for hit in hits:
        snippet = hit.payload.get("snippet", "")
        url = hit.payload.get("url", "")
        context_texts.append(snippet)
        if url:
            links.append({"url": url, "text": snippet[:80] + "..."})

    # Prepare prompt for Ollama LLM
    prompt = f"Answer the question based on context:\n\nContext:\n" + "\n---\n".join(context_texts) + f"\n\nQuestion: {query.question}\nAnswer:"

    # Call Ollama
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{OLLAMA_URL}/v1/llm/llama", json={"prompt": prompt, "model": "llama-3"})
        response.raise_for_status()
        data = response.json()

    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "Sorry, no answer generated.")

    return {"answer": answer, "links": links}
