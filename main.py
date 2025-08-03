import streamlit as st
from utility import process_document_to_faiss_db, answer_question

st.set_page_config(page_title="DOC_SUMZ - RAG QA", layout="centered")
st.title("📄 DOC_SUMZ: Ask Your PDF!")

uploaded_pdf = st.file_uploader("Upload your PDF file", type=["pdf"])
query = st.text_input("Enter your question:")

if uploaded_pdf and query:
    with st.spinner("Processing..."):
        db = process_document_to_faiss_db(uploaded_pdf)
        response = answer_question(db, query)
    st.markdown("### 💡 Answer:")
    st.success(response)
