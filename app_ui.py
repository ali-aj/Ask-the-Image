import streamlit as st
from PIL import Image
from typing import Optional
from audio_recorder_streamlit import audio_recorder
from config import (
    MAX_RECORDING_DURATION,
    APP_TITLE,
    SUPPORTED_IMAGE_TYPES
)

class AppUI:
    def __init__(self, services: dict):
        self.services = services
        self._init_session_state()
    
    def _init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    def _render_sidebar(self) -> Optional[Image.Image]:
        with st.sidebar:
            st.header("Controls")
            image_source = st.radio("Image Source", ["Upload", "Camera"])
            return self._handle_image_upload() if image_source == "Upload" \
                else self._handle_camera_input()
    
    def _handle_image_upload(self) -> Optional[Image.Image]:
        image_file = st.file_uploader(
            "Upload Image", 
            type=SUPPORTED_IMAGE_TYPES
        )
        return Image.open(image_file) if image_file else None
    
    def _handle_camera_input(self) -> Optional[Image.Image]:
        image_capture = st.camera_input("Take a Photo")
        return Image.open(image_capture) if image_capture else None
    
    def _render_qa_history(self):
        st.subheader("Q&A History")
        for msg in reversed(st.session_state.messages):
            with st.expander(f"Q: {msg['question']}"):
                st.image(msg["image"], width=200)
                st.write(f"A: {msg['answer']}")
                if st.button("🔊 Play", key=msg['answer']):
                    self.services['tts'].speak(msg["answer"])
    
    def run(self):
        st.title(APP_TITLE)
        image = self._render_sidebar()
        
        if image:
            self._render_image_and_process_audio(image)
        else:
            st.info("Please upload or capture an image to begin")
    
    def _render_image_and_process_audio(self, image: Image.Image):
        st.image(image, use_column_width=True)
        audio_bytes = audio_recorder(
            "Record Question", 
            pause_threshold=MAX_RECORDING_DURATION,
            sample_rate=16_000
        )
        
        if audio_bytes:
            self._process_audio_input(image, audio_bytes)
    
    def _process_audio_input(self, image: Image.Image, audio_bytes: bytes):
        question = self.services['stt'].transcribe_audio(audio_bytes)
        if question:
            answer = self.services['vqa'].generate_answer(image, question)
            if answer:
                self._update_session_state(image, question, answer)
                self._render_qa_history()
    
    def _update_session_state(self, image: Image.Image, question: str, answer: str):
        st.session_state.messages.append({
            "question": question,
            "answer": answer,
            "image": image
        })