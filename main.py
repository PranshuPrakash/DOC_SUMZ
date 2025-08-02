import os
import re
import streamlit as st
from dotenv import load_dotenv

from utility import process_document_to_chroma_db, get_answer_with_llm

from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit app config
st.set_page_config(page_title="DOC_SUMZ QA-RAG", layout="centered")
st.title("📄 DOC_SUMZ — Multi-LLM RAG")

# Sidebar: LLM selector
st.sidebar.header("🧠 Choose LLM")
model_choice = st.sidebar.selectbox(
    "Select Language Model",
    ["DeepSeek-R1", "GPT-3.5", "Claude 3 Sonnet", "Gemini-Pro"]
)

# Sidebar: File uploader
st.sidebar.header("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Handle uploaded file
if uploaded_file is not None:
    working_dir = os.getcwd()
    save_path = os.path.join(working_dir, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document_to_chroma_db(uploaded_file.name)
    st.sidebar.success("✅ Document Processed Successfully!")

# Question input
user_question = st.text_input("Ask a question about the document:")

# Get answer on button click
if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        # Choose the appropriate LLM
        if model_choice == "GPT-3.5":
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        elif model_choice == "Claude 3 Sonnet":
            llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
        elif model_choice == "Gemini-Pro":
            llm = ChatGoogleGenerativeAI(model="gemini-pro")
        else:  # Default to DeepSeek
            llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)

        # Get the answer
        raw_answer = get_answer_with_llm(user_question, llm)

        # Clean up the response
        filtered_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

        # Display answer
        st.markdown("### 🤖 Response")
        st.markdown(filtered_answer)
