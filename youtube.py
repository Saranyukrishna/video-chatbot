import streamlit as st
import requests

# Vimeo API token (replace this with your token)
ACCESS_TOKEN = "4d5aaca6ab88028a924d29c68d87a46b"

# Function to get the transcript (captions) for a given video ID
def get_vimeo_transcript(video_id: str):
    url = f"https://api.vimeo.com/videos/{video_id}/texttracks"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        texttracks = response.json()
        if texttracks["data"]:
            # Assuming English subtitles, you can modify this if needed
            for track in texttracks["data"]:
                if track["language"] == "en":
                    caption_url = track["link"]
                    # Fetch the caption file (subtitles)
                    caption_response = requests.get(caption_url)
                    if caption_response.status_code == 200:
                        return caption_response.text
                    else:
                        return "Error: Unable to fetch captions file."
        else:
            return "Error: No captions available for this video."
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit UI
st.title("Vimeo Transcript Extractor")

video_url = st.text_input("Enter Vimeo video URL:")

if video_url:
    # Extract the video ID from the Vimeo URL
    if "vimeo.com/" in video_url:
        video_id = video_url.split("vimeo.com/")[-1]
    else:
        st.error("Invalid Vimeo URL format.")
        st.stop()
    
    if st.button("Get Transcript"):
        with st.spinner("Fetching transcript..."):
            transcript = get_vimeo_transcript(video_id)
            if transcript.startswith("Error"):
                st.error(transcript)
            else:
                st.success("Transcript fetched successfully!")
                st.text_area("Transcript:", transcript, height=400)
