from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class RAGRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def index_documents(self, documents: List[str]):
        """Index documents for similarity search"""
        self.documents = documents
        embeddings = self.model.encode(documents)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, any]]:
        """Retrieve top-k relevant documents for the query"""
        if self.index is None or not self.documents:
            return []

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), k
        )

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({"content": self.documents[idx], "score": float(score)})
        return results
