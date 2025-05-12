import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from urllib.parse import urlparse, parse_qs

st.title("YouTube Transcript Extractor (Auto + Manual Captions)")

video_url = st.text_input("Enter YouTube Video URL:")

def extract_video_id(url):
    parsed_url = urlparse(url)
    return parse_qs(parsed_url.query).get("v", [None])[0]

if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                transcript = transcript_list.find_transcript(['en'])
            except NoTranscriptFound:
                transcript = transcript_list.find_generated_transcript(['en'])

            result = transcript.fetch()
            full_text = "\n".join([entry["text"] for entry in result])
            st.subheader("Transcript:")
            st.text(full_text)

        except TranscriptsDisabled:
            st.warning("Subtitles are disabled for this video.")
        except NoTranscriptFound:
            st.warning("No English subtitles found (manual or auto-generated).")
        except CouldNotRetrieveTranscript:
            st.warning("Could not retrieve subtitles. Video may be restricted.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Invalid YouTube URL.")
