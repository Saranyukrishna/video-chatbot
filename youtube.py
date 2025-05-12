import streamlit as st
import whisper
import os
import tempfile
import subprocess

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

st.title("Saranyu - YouTube QA via Whisper")

video_url = st.text_input("Enter YouTube Video URL:")
question = st.text_area("Ask a question based on the video:")

def download_audio(video_url):
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.mp3")
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "-o", audio_path,
        video_url
    ]
    try:
        subprocess.run(command, check=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        st.error(f"Audio download failed: {str(e)}")
        raise

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

if st.button("Get Answer"):
    if not video_url or not question:
        st.warning("Please provide both video URL and a question.")
    else:
        try:
            st.info("Downloading and transcribing audio...")
            audio_file = download_audio(video_url)
            transcript = transcribe_audio(audio_file)

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

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
