import os
import re
import streamlit as st
from utility import process_document_to_chroma_db, answer_question

# Set page configuration
st.set_page_config(page_title="DOC_SUMZ - Ask Your PDF", layout="centered")

# Main title
st.title("📄 DOC_SUMZ: Ask Questions from Your PDF")

# Sidebar
st.sidebar.title("📂 Upload Section")
st.sidebar.markdown("Upload a **PDF document** and ask questions from it using advanced RAG with LLM.")

uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

# Flags
doc_loaded = False

# If PDF uploaded
if uploaded_file is not None:
    working_dir = os.getcwd()
    save_path = os.path.join(working_dir, uploaded_file.name)

    with st.spinner("Processing your document... ⏳"):
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        process_document_to_chroma_db(uploaded_file.name)
        doc_loaded = True

    st.sidebar.success("✅ Document Processed and Ready!")

# User input section
st.markdown("## 💬 Ask Your Question")
user_question = st.text_input("Type your question below and hit **Enter** or click the button:")

if st.button("🔍 Get Answer"):
    if not uploaded_file:
        st.warning("⚠️ Please upload a document first.")
    elif not user_question.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("Generating answer... 🤖"):
            raw_answer = answer_question(user_question)
            filtered_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

        st.markdown("### 🧠 DeepSeek-R1's Answer")
        st.success(filtered_answer)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + RAG + Groq")
