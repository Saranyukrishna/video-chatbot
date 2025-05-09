import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

st.title("saranyu")

video_id = st.text_input("Enter YouTube Video ID (e.g. Gfr50f6ZBvo):")
question = st.text_area("Ask a question based on the transcript:")

proxy_list = [
    "http://45.91.133.137:8080",
    "http://185.199.229.156:7492",
    "http://194.67.91.153:3128",
    "http://82.165.184.53:80",
    "http://149.56.96.252:9300"
]

def get_transcript_with_proxies(video_id):
    for proxy in proxy_list:
        try:
            proxies = {"http": proxy, "https": proxy}
            return YouTubeTranscriptApi.get_transcript(video_id, languages=["en"], proxies=proxies)
        except Exception:
            continue
    raise Exception("All proxies failed or YouTube blocked access.")

if st.button("Get Answer"):
    if not video_id or not question:
        st.warning("Please provide both video ID and a question.")
    else:
        try:
            transcript_list = get_transcript_with_proxies(video_id)
            transcript = " ".join(chunk["text"] for chunk in transcript_list)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            retrieved_docs = retriever.invoke(question)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=["context", "question"]
            )

            llm = ChatGroq(
                model="llama3-70b-8192",
                temperature=1.5
            )

            final_prompt = prompt.invoke({"context": context_text, "question": question})
            answer = llm.invoke(final_prompt)

            st.subheader("Answer:")
            st.write(answer.content)

        except TranscriptsDisabled:
            st.error("No captions available for this video.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
