# Patch to use modern SQLite from pysqlite3 (for Streamlit Cloud compatibility)
import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3
sys.modules["sqlite3.dbapi2"] = pysqlite3

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


working_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# loading the embedding model
embedding = HuggingFaceEmbeddings()

# load the llm form groq
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0
)


import streamlit as st

def process_document_to_chroma_db(file_name):
    try:
        # 1️⃣ Load the PDF
        loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
        documents = loader.load()

        if not documents:
            st.error("❌ No text could be extracted from the uploaded PDF. Please check the file.")
            return

        # 2️⃣ Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        if not texts:
            st.error("❌ Document loaded, but no text chunks were created after splitting.")
            return

        # 3️⃣ Test embeddings
        try:
            test_vec = embedding.embed_query("Test embedding")
            if not test_vec or len(test_vec) == 0:
                st.error("❌ Embedding model returned empty results. Check model name or internet connection.")
                return
        except Exception as e:
            st.error(f"❌ Failed to initialize embeddings: {e}")
            return

        # 4️⃣ Save to Chroma DB
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory=f"{working_dir}/doc_vectorstore"
        )

        st.sidebar.success("✅ Document processed and stored successfully!")

    except Exception as e:
        st.error(f"⚠️ Unexpected error while processing document: {e}")


def answer_question(user_question):
    # load the persistent vectordb
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    # retriever
    retriever = vectordb.as_retriever()

    # create a chain to answer user question usinng DeepSeek-R1
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer


