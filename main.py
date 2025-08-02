import os
import re
import streamlit as st
from utility import process_document_to_chroma_db, answer_question

# Page config
st.set_page_config(page_title="DOC_SUMZ - Free RAG", layout="centered")
st.title("📄 DOC_SUMZ: RAG with Free Local LLM")

# Sidebar file upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    file_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    process_document_to_chroma_db(file_path)
    st.sidebar.success("✅ Document Indexed")

# Question interface
user_question = st.text_input("Ask something from the document:")

if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        raw_answer = answer_question(user_question)
        cleaned_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
        st.markdown("### 🧠 DeepSeek-R1 Answer")
        st.markdown(cleaned_answer)
