import faiss
import numpy as np


class FAISSRetriever:
    def __init__(self, embeddings, metadata):
        self.embeddings = np.array(embeddings).astype("float32")
        self.metadata = metadata

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query_embedding, top_k=5, filter_type=None):
        query = np.array(query_embedding).astype("float32")
        distances, indices = self.index.search(query, top_k)

        results = []
        scores = []

        for idx, dist in zip(indices[0], distances[0]):
            if filter_type:
                if self.metadata[idx]["type"] != filter_type:
                    continue

            results.append(idx)
            scores.append(float(1 / (1 + dist)))  # convert to similarity

        return results, scores
