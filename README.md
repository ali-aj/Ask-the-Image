# Ask-the-Image Mini App

A Streamlit-based application that lets users upload or capture an image, ask a question via speech, and receive a text answer (with optional text-to-speech playback) powered by Whisper ASR and BLIP-2 visual-language models.

## Repository Structure

- **app.py**  
  Main entry point: initializes services and launches the Streamlit UI.
- **app_ui.py**  
  Defines the Streamlit interface (`AppUI`), handling image input, audio recording, and Q&A history.
- **model_loader.py**  
  Loads and caches Whisper ASR and BLIP-2 components via `ModelLoader`.
- **speech_to_text.py**  
  `SpeechToTextService`: records/transcribes audio using Whisper.
- **image_qa.py**  
  `ImageQAService`: generates answers to image-based questions via BLIP-2.
- **text_to_speech.py**  
  `TextToSpeechService`: plays answers aloud using `pyttsx3`.
- **config.py**  
  Global constants and model configuration.
- **requirements.txt**  
  Python dependencies.
- **.gitignore**  
  Excludes virtual environment and cache directories.

## Installation

1. Clone the repository and enter the directory:  
   ```bash
   git clone https://github.com/ali-aj/Ask-the-Image
   cd Ask-the-Image
   ```

2. Create and activate a virtual environment:  
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   # or
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the Streamlit app:
```bash
streamlit run app.py
```

1. In the sidebar, choose **Upload** or **Camera** to provide an image.  
2. Record your question (max duration defined by `MAX_RECORDING_DURATION` in `config.py`).  
3. View the generated answer on-screen and click 🔊 to replay via text-to-speech.

## Configuration

All settings live in `config.py`:
```python
MODEL_CONFIG = {
    "asr_model": "small",
    "vlm_processor": "Salesforce/blip2-opt-2.7b",
    "vlm_model": "Salesforce/blip2-opt-2.7b"
}
MAX_RECORDING_DURATION = 10.0
APP_TITLE = "Ask-the-Image Mini App"
SUPPORTED_IMAGE_TYPES = ["jpg", "png", "jpeg"]
```

## Services Overview

- **ModelLoader.load_asr_model** & **load_vlm_components**  
  Load Whisper ASR and BLIP-2 processor/model.
- **SpeechToTextService.transcribe_audio**  
  Transcribe recorded audio bytes to text.
- **ImageQAService.generate_answer**  
  Generate an answer given a PIL Image and question text.
- **TextToSpeechService.speak**  
  Play a given text answer using `pyttsx3`.
- **AppUI.run**  
  Render the Streamlit interface and orchestrate image upload, audio Q&A, and response playback.