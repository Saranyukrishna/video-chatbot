import streamlit as st
from pytube import YouTube

st.title("YouTube Transcript Extractor using Pytube")

video_url = st.text_input("Enter YouTube video URL:")

if video_url:
    try:
        yt = YouTube(video_url)
        caption = yt.captions.get_by_language_code('en')
        if caption:
            transcript = caption.generate_srt_captions()
            st.subheader("Transcript (SRT Format):")
            st.text(transcript)
        else:
            st.error("No English captions available for this video.")
    except Exception as e:
        st.error(f"Error: {e}")
