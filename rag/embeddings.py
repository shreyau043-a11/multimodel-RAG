import requests
from config import JINA_MODEL, JINA_EMBEDDING_URL


def get_jina_embeddings(texts, jina_key):
    headers = {
        "Authorization": f"Bearer {jina_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": JINA_MODEL,
        "input": texts
    }

    response = requests.post(
        JINA_EMBEDDING_URL,
        headers=headers,
        json=payload
    )

    response.raise_for_status()

    data = response.json()["data"]
    return [item["embedding"] for item in data]
