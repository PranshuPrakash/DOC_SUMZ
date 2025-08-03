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


def process_document_to_chroma_db(file_name):
    # load the doc using unstructured
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    # splitting te text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0


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

