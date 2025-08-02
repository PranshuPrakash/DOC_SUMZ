import os
import re
import streamlit as st
from utility import process_document_to_chroma_db, answer_question

# Set page configuration
st.set_page_config(page_title="DOC_SUMZ - RAG QA", layout="centered")
st.title("📄 DOC_SUMZ - RAG Based QA System")

# Sidebar: File uploader
st.sidebar.header("📤 Upload Your PDF Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file locally
    file_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process and index the document
    process_document_to_chroma_db(file_path)
    st.sidebar.success("✅ Document indexed successfully!")

# User input
user_question = st.text_input("❓ Ask a question based on the document:")

if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        raw_answer = answer_question(user_question)
        clean_answer = re.sub(r"<.*?>", "", raw_answer).strip()

        st.markdown("### 🤖 Re
