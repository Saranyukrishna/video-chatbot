import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript

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

st.title("📄 YouTube Transcript Extractor (No Download)")

video_url = st.text_input("Enter YouTube video URL:")
if video_url:
    if "watch?v=" in video_url:
        video_id = video_url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[-1].split("?")[0]
    else:
        st.error("Invalid YouTube URL format.")
        st.stop()

    if st.button("Get Transcript"):
        with st.spinner("Fetching transcript..."):
            transcript = get_transcript(video_id)
            if transcript.startswith("[ERROR]"):
                st.error(transcript)
            else:
                st.success("Transcript fetched successfully!")
                st.text_area("Transcript:", transcript, height=400)
