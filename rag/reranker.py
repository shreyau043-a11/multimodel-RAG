import numpy as np
from rag.embeddings import get_text_embedding

def rerank(query, documents):
    query_vec = get_text_embedding(query)
    scores = []

    for doc in documents:
        doc_vec = get_text_embedding(doc)
        score = np.dot(query_vec, doc_vec)
        scores.append((doc, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scores]

       
