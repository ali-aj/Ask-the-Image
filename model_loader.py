import torch
import whisper
import streamlit as st
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from config import MODEL_CONFIG

class ModelLoader:
    @staticmethod
    @st.cache_resource
    def load_asr_model():
        return whisper.load_model(MODEL_CONFIG["asr_model"])
    
    @staticmethod
    @st.cache_resource
    def load_vlm_components():
        processor = Blip2Processor.from_pretrained(MODEL_CONFIG["vlm_processor"])
        model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_CONFIG["vlm_model"], 
            torch_dtype=torch.float16
        )
        return processor, model