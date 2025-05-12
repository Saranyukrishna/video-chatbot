import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

st.title("📄 YouTube Transcript Extractor and Question Answering")

# Function to get transcript from YouTube
def get_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(['en'])

        fetched = transcript.fetch()
        return "\n".join([entry['text'] for entry in fetched])

    except TranscriptsDisabled:
        return "[ERROR] Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "[ERROR] No English transcript found (manual or auto-generated)."
    except CouldNotRetrieveTranscript:
        return "[ERROR] Could not retrieve transcript (possibly restricted or region-locked)."
    except Exception as e:
        return f"[ERROR] Unexpected error: {e}"

# Input fields for video URL and question
video_url = st.text_input("Enter YouTube video URL:")
question = st.text_area("Ask a question based on the transcript:")

if video_url:
    if "watch?v=" in video_url:
        video_id = video_url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[-1].split("?")[0]
    else:
        st.error("Invalid YouTube URL format.")
        st.stop()

    if st.button("Get Transcript and Answer Question"):
        with st.spinner("Fetching transcript and answering question..."):
            # Fetch transcript
            transcript = get_transcript(video_id)
            
            if transcript.startswith("[ERROR]"):
                st.error(transcript)
            else:
                st.success("Transcript fetched successfully!")

                # Split transcript into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

                # Generate embeddings and vector store
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                # Retrieve documents based on question
                retrieved_docs = retriever.invoke(question)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

                # Generate prompt for answering
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

                # Use Groq-based LLM to get the answer
                llm = ChatGroq(
                    model="llama3-70b-8192",
                    temperature=1.5
                )

                final_prompt = prompt.invoke({"context": context_text, "question": question})
                answer = llm.invoke(final_prompt)

                st.subheader("Answer:")
                st.write(answer.content)
