"""
use google speech instead of deep speech
"""

import logging
import logging.handlers
import queue
import threading
import time
from pathlib import Path
from collections import deque
from typing import List

import av
import numpy as np
import pydub
import streamlit as st
import speech_recognition as sr

from streamlit_webrtc import WebRtcMode, webrtc_streamer

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

recognizer = sr.Recognizer()

def start_recording(recording_in_progress):
    user_output = ""
    print("Recording started")

    # Infinite loop for continuous recording
    with sr.Microphone() as source:
        if recording_in_progress:
            try:
                # Listen for the specified chunk duration
                audio = recognizer.listen(source, timeout=5)
                recognized_text = recognizer.recognize_google(audio)
                st.text(" - " + recognized_text)
                user_output += recognized_text + " "
            except sr.UnknownValueError:
                st.text("Could not understand audio")
            except sr.WaitTimeoutError:
                st.text("Timeout. No speech detected in the last 5 seconds")

    return user_output

def main():
    st.header("Real Time Speech-to-Text")
    st.markdown(
        """
This demo app is using Google SpeechRecognition, a cloud-based speech-to-text service.
"""
    )

    app_sst()


def app_sst():
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.audio_receiver:
            if stream is None:
                stream = webrtc_ctx.audio_receiver
                status_indicator.write("Audio stream connected.")

            # Call your speech recognition function here
            recognized_text = start_recording(webrtc_ctx.state.playing)

            if recognized_text:
                text_output.markdown(f"**Text:** {recognized_text}")
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
