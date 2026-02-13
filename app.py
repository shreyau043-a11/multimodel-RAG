import streamlit as st
from rag.chunking import chunk_text
from rag.retriever import VectorStore
from rag.reranker import rerank
from rag.llm import generate_answer
from rag.vision import get_image_embedding
import config

st.title("Multimodal RAG System")

uploaded_file = st.file_uploader("Upload Text File", type=["txt"])
image_file = st.file_uploader("Upload Image", type=["jpg", "png"])

query = st.text_input("Enter your question")

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(text, 500)

    vector_store = VectorStore()
    vector_store.add_texts(chunks)

    st.success("Document processed!")


if st.button("Ask"):
    if query:
        retrieved_docs = vector_store.search(query, config.TOP_K)
        reranked_docs = rerank(query, retrieved_docs)
        answer = generate_answer(query, reranked_docs)

        st.subheader("Answer")
        st.write(answer)

if image_file:
    image_embedding = get_image_embedding(image_file)
    st.success("Image embedding created!")
   
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(text, 500)

    vector_store = VectorStore()

    vector_store.add_texts(chunks)

    st.success("Document processed!")


    


