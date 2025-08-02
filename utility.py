import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Set working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Free embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Free HuggingFace-hosted LLM
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.1, "max_length": 512})

def process_document_to_chroma_db(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=os.path.join(working_dir, "doc_vectorstore")
    )
    vectordb.persist()
    return True

def answer_question(question):
    vectordb = Chroma(
        persist_directory=os.path.join(working_dir, "doc_vectorstore"),
        embedding_function=embedding
    )
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    result = qa_chain.invoke({"query": question})
    return result["result"]
