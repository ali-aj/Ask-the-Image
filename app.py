from model_loader import ModelLoader
from speech_to_text import SpeechToTextService
from image_qa import ImageQAService
from text_to_speech import TextToSpeechService
from app_ui import AppUI

def setup_services():
    """Initialize and return all services"""
    processor, model = ModelLoader.load_vlm_components()
    return {
        'stt': SpeechToTextService(ModelLoader.load_asr_model()),
        'vqa': ImageQAService(processor, model),
        'tts': TextToSpeechService()
    }

def main():
    """Main application setup and execution"""
    services = setup_services()
    app_ui = AppUI(services)
    app_ui.run()

if __name__ == "__main__":
    main()