import streamlit as st
import speech_recognition as sr

# Create a SpeechRecognition recognizer
recognizer = sr.Recognizer()

def main():
    st.title("Real Time Speech-to-Text with Google Recognition")

    # Start recording button
    recording_in_progress = st.button("Start Recording")

    # If recording is in progress, capture and save audio
    if recording_in_progress:
        captured_audio = capture_audio()
        save_audio_to_file(captured_audio)

        # Display the saved audio
        st.audio(captured_audio.get_wav_data(), format='audio/wav')

        # Perform speech recognition on the saved audio
        recognized_text = recognize_audio(captured_audio)
        st.header("Recognized Text:")
        st.write(recognized_text)

def capture_audio():
    with sr.Microphone() as source:
        st.info("Recording... Speak something!")
        audio_data = recognizer.listen(source, timeout=10)
        st.success("Recording complete!")
    return audio_data

def save_audio_to_file(audio_data, file_path="output_audio.wav"):
    with open(file_path, "wb") as audio_file:
        audio_file.write(audio_data.get_wav_data())

def recognize_audio(audio_data):
    try:
        recognized_text = recognizer.recognize_google(audio_data)
        return recognized_text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

if __name__ == "__main__":
    main()
