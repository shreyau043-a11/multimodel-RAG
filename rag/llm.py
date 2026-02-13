import os
from groq import Groq

# Get API key from environment (Streamlit Secrets)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment")

client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query, context_docs):
    if not context_docs:
        context = "No relevant documents found."
    else:
        context = "\n\n".join(context_docs)

    prompt = f"""
You are a helpful assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"GROQ ERROR: {str(e)}"
