import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

st.title("YouTube Transcript Extractor")

video_url = st.text_input("Enter YouTube Video URL:")

def extract_video_id(url):
    parsed_url = urlparse(url)
    return parse_qs(parsed_url.query).get("v", [None])[0]

if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            full_text = "\n".join([entry["text"] for entry in transcript])
            st.subheader("Transcript:")
            st.text(full_text)
        except Exception as e:
            st.error(f"Transcript not available: {e}")
    else:
        st.error("Invalid YouTube URL format.")
