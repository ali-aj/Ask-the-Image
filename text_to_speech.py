import pyttsx3
import streamlit as st

class TextToSpeechService:
    @staticmethod
    def speak(text: str) -> None:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            st.error(f"TTS error: {str(e)}")