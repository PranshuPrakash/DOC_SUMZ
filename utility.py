import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

def process_document_to_faiss_db(pdf_file):
    temp_path = f"temp_{pdf_file.name}"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.read())

    loader = PyMuPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    os.remove(temp_path)
    return db

def answer_question(db, query):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.3, "max_length": 256}
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = db.similarity_search(query)
    return chain.run(input_documents=docs, question=query)
