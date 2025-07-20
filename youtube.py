import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()
qa_chain = None

class TranscriptInput(BaseModel):
    transcript: str

class QuestionInput(BaseModel):
    question: str

@app.post("/upload-transcript")
def upload(data: TranscriptInput):
    global qa_chain
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([data.transcript])
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embed)
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vs.as_retriever())
    return {"status": "ok"}

@app.post("/ask")
def ask(data: QuestionInput):
    if not qa_chain:
        return {"error": "Transcript not loaded."}
    return {"answer": qa_chain.run(data.question)}
