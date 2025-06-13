import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Load data
with open("data/discourse_posts.json", "r") as f:
    posts = json.load(f)

# Initialize model & client
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(host="localhost", port=6333)

collection_name = "tds_posts"
qdrant.recreate_collection(collection_name, vector_size=384, distance="Cosine")

points = []
for post in posts:
    text = post.get("content", "")  # Adjust key based on JSON structure
    post_id = post.get("id")
    url = post.get("url", "")
    snippet = text[:200]

    embedding = model.encode(text).tolist()
    point = PointStruct(id=post_id, vector=embedding, payload={"url": url, "snippet": snippet})
    points.append(point)

qdrant.upsert(collection_name=collection_name, points=points)
print(f"Uploaded {len(points)} posts to Qdrant.")
