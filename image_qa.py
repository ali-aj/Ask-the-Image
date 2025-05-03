from PIL import Image
import torch
import streamlit as st
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Optional

class ImageQAService:
    def __init__(self, processor: Blip2Processor, model: Blip2ForConditionalGeneration):
        self.processor = processor
        self.model = model
    
    def generate_answer(self, image: Image.Image, question: str) -> Optional[str]:
        try:
            inputs = self.processor(
                images=image, 
                text=question, 
                return_tensors="pt"
            ).to(torch.float16)
            
            generated_ids = self.model.generate(**inputs)
            return self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
        except Exception as e:
            st.error(f"QA Generation error: {str(e)}")
            return None