import sys, os
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
sys.modules["sqlite3.dbapi2"] = pysqlite3
from chromadb import Client import os
import re
import streamlit as st
from utility import process_document_to_chroma_db, answer_question

# Set page configuration
st.set_page_config(page_title="QA-RAG", layout="centered")

st.title("QA Document RAG")

# Sidebar for file upload
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save file
    working_dir = os.getcwd()
    save_path = os.path.join(working_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document_to_chroma_db(uploaded_file.name)
    st.sidebar.success("✅ Document Processed Successfully!")

# User input
user_question = st.text_input("Ask a question about the document:")

if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        raw_answer = answer_question(user_question)

        # Remove <think> </think> and its content
        filtered_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

        st.markdown("### 🤖 DeepSeek-R1 Response")
        st.markdown(filtered_answer)

