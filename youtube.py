import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtubesearchpython import VideosSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

st.title("Ask Questions from YouTube Video Transcript")

query = st.text_input("Enter YouTube video search query:")
question = st.text_area("Ask a question based on the transcript:")

if st.button("Search and Answer"):
    if not query or not question:
        st.warning("Please provide both a search query and a question.")
    else:
        try:
            search = VideosSearch(query, limit=1)
            result = search.result()
            video_id = result["result"][0]["id"]
            video_title = result["result"][0]["title"]

            st.write(f"**Video Found:** {video_title}")
            st.video(f"https://www.youtube.com/watch?v={video_id}")

            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
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
