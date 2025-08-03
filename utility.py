import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings


def process_document_to_faiss_db(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local("faiss_index")

def answer_question(query: str) -> str:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 200}
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    result = qa_chain.run(query)
    return result

