from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class VectorStore:
    def __init__(self, vector_size=384):
        self.client = QdrantClient(":memory:")
        self.collection = "biomedical"

        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    def add_documents(self, embeddings, texts):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={"text": texts[i]}
            )
            for i in range(len(texts))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    # ✅ FIXED HERE (NEW QDRANT API)
    def search(self, query_vector, top_k=5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k
        )

        return [point.payload["text"] for point in results.points]