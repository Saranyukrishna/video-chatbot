import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import VideosSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="YouTube AI Assistant", page_icon="▶️", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #f5f5f5;
}
.stSidebar {
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
}
.stChatInput {
    bottom: 20px;
    position: fixed;
    width: 70%;
}
.stChatMessage {
    border-radius: 15px;
    padding: 12px 15px;
    margin: 8px 0;
}
.stChatMessage.user {
    background-color: #e3f2fd;
    margin-left: auto;
    max-width: 80%;
}
.stChatMessage.assistant {
    background-color: #f1f1f1;
    margin-right: auto;
    max-width: 80%;
}
.stButton button {
    background-color: #ff0000;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 8px 15px;
}
.stButton button:hover {
    background-color: #cc0000;
}
.stRadio div[role="radiogroup"] {
    background-color: white;
    padding: 10px;
    border-radius: 5px;
}
.stTextInput input {
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=100)
with col2:
    st.title("YouTube AI Assistant")
    st.caption("Ask anything about any YouTube video")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.video_info = None

with st.sidebar:
    st.header("Video Input")
    input_method = st.radio("Select input method:", ["Search", "URL"])
    
    if input_method == "Search":
        search_query = st.text_input("Search YouTube videos")
        if st.button("Find Video"):
            with st.spinner("Searching..."):
                try:
                    search = VideosSearch(search_query, limit=1)
                    result = search.result()
                    video_id = result["result"][0]["id"]
                    video_title = result["result"][0]["title"]
                    st.session_state.video_info = {
                        "id": video_id,
                        "title": video_title,
                        "url": f"https://www.youtube.com/watch?v={video_id}"
                    }
                    st.success("Video found!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    else:
        video_url = st.text_input("Paste YouTube URL")
        if st.button("Load Video"):
            with st.spinner("Processing..."):
                try:
                    video_id = video_url.split("v=")[-1].split("&")[0]
                    search = VideosSearch(video_id, limit=1)
                    result = search.result()
                    video_title = result["result"][0]["title"]
                    st.session_state.video_info = {
                        "id": video_id,
                        "title": video_title,
                        "url": video_url
                    }
                    st.success("Video loaded!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.video_info:
        st.divider()
        st.subheader("Current Video")
        st.video(st.session_state.video_info["url"])
        st.caption(st.session_state.video_info["title"])
        
        if st.button("Clear Video"):
            st.session_state.messages = []
            st.session_state.video_info = None
            st.rerun()

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if st.session_state.video_info and not any(m["role"] == "assistant" and "transcript" in m["content"].lower() for m in st.session_state.messages):
    with st.spinner("Processing video transcript..."):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(st.session_state.video_info["id"], languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"✅ Transcript loaded for: {st.session_state.video_info['title']}\n\nYou can now ask questions about this video!"
            })
            st.rerun()
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"❌ Error loading transcript: {str(e)}"
            })
            st.rerun()

if prompt := st.chat_input("Ask about the video..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    with st.spinner("Thinking..."):
        try:
            retrieved_docs = st.session_state.retriever.invoke(prompt)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
            
            prompt_template = ChatPromptTemplate.from_template("""
            You're an expert YouTube assistant. Answer the question based on the video transcript.
            Be concise but helpful. If unsure, say you don't know.
            
            Transcript excerpt: {context}
            
            Question: {question}
            
            Answer:
            """)
            
            chain = prompt_template | llm
            response = chain.invoke({"context": context_text, "question": prompt})
            
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response.content)
        
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"❌ Error answering question: {str(e)}"
            })
            with chat_container:
                with st.chat_message("assistant"):
                    st.error(f"Error: {str(e)}")

if st.session_state.video_info:
    with st.sidebar:
        st.divider()
        st.subheader("Quick Actions")
        
        if st.button("Summarize Video"):
            with st.spinner("Generating summary..."):
                try:
                    llm = ChatGroq(model="llama3-70b-8192", temperature=0.5)
                    prompt = ChatPromptTemplate.from_template("""
                    Create a concise 3-5 bullet point summary of this YouTube video transcript:
                    
                    {transcript}
                    
                    Summary:
                    """)
                    chain = prompt | llm
                    summary = chain.invoke({"transcript": transcript[:5000]})
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"📝 **Video Summary**\n\n{summary.content}"
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.button("Key Topics"):
            with st.spinner("Identifying topics..."):
                try:
                    llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
                    prompt = ChatPromptTemplate.from_template("""
                    List 5-7 main topics covered in this video transcript:
                    
                    {transcript}
                    
                    Topics:
                    """)
                    chain = prompt | llm
                    topics = chain.invoke({"transcript": transcript[:5000]})
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"🏷️ **Key Topics**\n\n{topics.content}"
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
