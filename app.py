import streamlit as st
import tempfile
import os

st.title("Multi-Object Tracking System")

st.write("Upload a video and track objects with unique IDs.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.write("Processing... ⏳")

    from run import run_tracker

    output_path = run_tracker(tfile.name)

    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()

    st.write("File size:", os.path.getsize(output_path))

    st.video(output_path)
    