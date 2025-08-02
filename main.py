import os
import re
import streamlit as st
from utility import process_document_to_faiss_db, answer_question

# Set page configuration
st.set_page_config(page_title="DOC_SUMZ - RAG QA", layout="centered")

st.title("DOC_SUMZ: Free RAG-based QA App")

# Sidebar for file upload
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save file
    working_dir = os.getcwd()
    save_path = os.path.join(working_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document_to_faiss_db(uploaded_file.name)
    st.sidebar.success("✅ Document Processed Successfully!")

# User input
user_question = st.text_input("Ask a question about the document:")

if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        raw_answer = answer_question(user_question)
        filtered_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

        st.markdown("### 🤖 Answer")
        st.markdown(filtered_answer)
