import requests
from config import GROQ_MODEL


def ask_llm(context, query, groq_key, model=None, temperature=0.3):
    model_name = model if model else GROQ_MODEL

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
