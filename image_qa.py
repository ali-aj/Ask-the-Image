from PIL import Image
import torch
import streamlit as st
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Optional

class ImageQAService:
    def __init__(self, processor: Blip2Processor, model: Blip2ForConditionalGeneration):
        self.processor = processor
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def generate_answer(self, image: Image.Image, question: str) -> Optional[str]:
        try:
            # Ensure image is in RGB mode
            image = image.convert('RGB')
            
            # Process inputs
            inputs = self.processor(
                images=image,
                text=f"Question: {question}\nAnswer:",  # Add explicit prompt format
                return_tensors="pt"
            )
            
            # Move inputs to model's device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate with controlled parameters
            with torch.cuda.amp.autocast():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=30,          # Reduce max tokens
                    min_new_tokens=2,           # Ensure minimum response length
                    num_beams=5,
                    early_stopping=True,        # Stop when complete
                    do_sample=True,             # Enable sampling
                    top_p=0.9,                 # Nucleus sampling
                    temperature=0.7,           # Add temperature
                    no_repeat_ngram_size=2     # Prevent repetition
                )
            
            # Decode the answer
            answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up the response
            answer = answer.replace("Question: ", "").replace(question, "").replace("Answer:", "").strip()
            
            return answer if answer else "I apologize, I couldn't generate a proper answer."
            
        except Exception as e:
            st.error(f"QA Generation error: {str(e)}")
            return None