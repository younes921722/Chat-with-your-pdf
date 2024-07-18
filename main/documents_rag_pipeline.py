from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

def get_pdf_text(pdf_docs):
    loader = PyPDFLoader(f"{pdf_docs}")
    docs = loader.load()
    return docs

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=20)
    chunks = text_splitter.split_text(docs)
    return chunks

def get_vectorstore(chunks):
    # Specifying the model path or name
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # model_name = "sentence-transformers/all-mpnet-base-v2"

    # Initializing the embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_db = FAISS.from_texts(texts=chunks,
                            embedding=embeddings
                            )
    print("*************************************************i passed the vector db")
    return vector_db

def get_conversation_chain(vectorstore):
    load_dotenv()

    sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

