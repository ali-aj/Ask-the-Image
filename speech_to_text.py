import io
import whisper
import streamlit as st
from typing import Optional

class SpeechToTextService:
    def __init__(self, asr_model: whisper.Whisper):
        self.asr_model = asr_model
    
    def transcribe_audio(self, audio_bytes: bytes) -> Optional[str]:
        try:
            audio = whisper.load_audio(io.BytesIO(audio_bytes))
            result = self.asr_model.transcribe(audio)
            return result["text"].strip()
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
            return None