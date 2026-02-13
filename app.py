import streamlit as st
import time
import json
from pypdf import PdfReader

from config import (
    GROQ_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
    MAX_HISTORY
)

from rag.embeddings import get_jina_embeddings
from rag.vision import describe_image
from rag.chunking import chunk_text
from rag.retriever import FAISSRetriever
from rag.reranker import simple_rerank
from rag.llm import ask_llm


st.set_page_config(page_title="Enterprise Multimodal RAG", layout="wide")
st.title("🚀 Enterprise Multimodal RAG Assistant")


if "history" not in st.session_state:
    st.session_state.history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []


with st.sidebar:
    st.header("⚙ Configuration")

    groq_key = st.text_input("Groq API Key", type="password")
    jina_key = st.text_input("Jina API Key", type="password")

    model = st.selectbox("LLM Model", [GROQ_MODEL])
    top_k = st.slider("Top K Retrieval", 1, 10, DEFAULT_TOP_K)
    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE)

    filter_type = st.radio("Scope", ["all", "text", "image"])

    if st.session_state.history:
        st.download_button(
            "Download Chat History",
            json.dumps(st.session_state.history, indent=2),
            file_name="chat_history.json"
        )


txt_files = st.file_uploader(
    "Upload TXT or PDF",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])


if txt_files and groq_key and jina_key:

    raw_text = ""

    for file in txt_files:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            raw_text += "\n".join(
                [p.extract_text() for p in reader.pages if p.extract_text()]
            )
        else:
            raw_text += file.read().decode("utf-8")

    chunks = chunk_text(raw_text)
    metadata = [{"type": "text"} for _ in chunks]

    if img_file:
        image_bytes = img_file.read()
        vision_text = describe_image(image_bytes, groq_key)
        if vision_text:
            chunks.append("Image description: " + vision_text)
            metadata.append({"type": "image"})

    embeddings = get_jina_embeddings(chunks, jina_key)
    retriever = FAISSRetriever(embeddings, metadata)

    st.session_state.retriever = retriever
    st.session_state.chunks = chunks


if st.session_state.retriever:

    query = st.text_input("Ask a question")

    if st.button("Generate Answer") and query:

        start = time.time()

        query_emb = get_jina_embeddings([query], jina_key)

        f = None if filter_type == "all" else filter_type

        ids, scores = st.session_state.retriever.search(
            query_emb,
            top_k=top_k,
            filter_type=f
        )

        retrieved_docs = [st.session_state.chunks[i] for i in ids]
        reranked = simple_rerank(query, retrieved_docs)
        context = "\n\n".join(reranked[:3])

        conversation_context = ""
        for q, a in st.session_state.history[-MAX_HISTORY:]:
            conversation_context += f"\nUser: {q}\nAssistant: {a}\n"

        final_context = conversation_context + "\n\n" + context

        answer = ask_llm(
            final_context,
            query,
            groq_key,
            model=model,
            temperature=temperature
        )

        st.session_state.history.append((query, answer))

        placeholder = st.empty()
        text_stream = ""
        for word in answer.split():
            text_stream += word + " "
            placeholder.markdown(text_stream)
            time.sleep(0.02)

        st.metric("Latency", round(time.time() - start, 2))

        with st.expander("Sources"):
            for idx in ids:
                st.write(st.session_state.chunks[idx][:300])
                st.divider()
