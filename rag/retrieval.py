import faiss
import numpy as np
from rag.embeddings import get_embeddings, get_text_embedding


class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []

    def add_texts(self, texts):
        embeddings = get_embeddings(texts)
        embeddings = np.array(embeddings).astype("float32")

        # Ensure 2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # 🔥 Automatically create index using embedding dimension
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query, top_k=3):
        query_vector = get_text_embedding(query)
        query_vector = np.array(query_vector).astype("float32")

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector, top_k)

        return [self.texts[i] for i in indices[0]]
