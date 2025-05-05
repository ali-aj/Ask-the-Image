import tempfile
import os
import whisper
import streamlit as st
from typing import Optional

class SpeechToTextService:
    def __init__(self, asr_model: whisper.Whisper):
        self.asr_model = asr_model
    
    def transcribe_audio(self, audio_bytes: bytes) -> Optional[str]:
        try:
            # Create temporary file to store audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio.flush()
                
                # Transcribe from temp file
                result = self.asr_model.transcribe(temp_audio.name)
                
            # Clean up temp file
            os.unlink(temp_audio.name)
            
            return result["text"].strip()
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
            return None