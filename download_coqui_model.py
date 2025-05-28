#!/usr/bin/env python3
"""
Script to download Coqui STT model for speech-to-text functionality.
"""

import os
import requests
from pathlib import Path
from utils.logger import logger

def download_file(url: str, destination: str) -> bool:
    """Download a file from URL to destination."""
    try:
        logger.info(f"Downloading {url} to {destination}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def download_coqui_stt_model():
    """Download the Coqui STT English model."""
    
    # Coqui STT English model URL (this is a smaller model for testing)
    model_url = "https://github.com/coqui-ai/STT-models/releases/download/english%2Fcoqui%2Fv1.0.0-huge-vocab/model.tflite"

    # Destination path
    model_path = "models/model.tflite"
    
    # Check if model already exists
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        return True
    
    logger.info("Downloading Coqui STT English model...")
    logger.info("This may take a few minutes depending on your internet connection.")
    
    success = download_file(model_url, model_path)
    
    if success:
        # Verify file size
        file_size = os.path.getsize(model_path)
        logger.info(f"Model downloaded successfully. Size: {file_size / (1024*1024):.1f} MB")
        
        # Also download the scorer file (optional but improves accuracy)
        scorer_url = "https://github.com/coqui-ai/STT-models/releases/download/english%2Fcoqui%2Fv1.0.0-huge-vocab/huge-vocabulary.scorer"
        scorer_path = "./models/coqui-stt-scorer.scorer"
        
        logger.info("Downloading language model scorer (optional, improves accuracy)...")
        download_file(scorer_url, scorer_path)
        
        return True
    else:
        logger.error("Failed to download Coqui STT model")
        return False

if __name__ == "__main__":
    print("Coqui STT Model Downloader")
    print("=" * 40)
    
    success = download_coqui_stt_model()
    
    if success:
        print("\n✅ Model download completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start the FastAPI server: python main.py")
        print("3. Test the /api/v1/stt/transcribe endpoint with a WAV file")
        print("\nAudio requirements:")
        print("- Format: WAV")
        print("- Sample rate: 16 kHz")
        print("- Channels: Mono")
        print("- Bit depth: 16-bit PCM")
    else:
        print("\n❌ Model download failed!")
        print("Please check your internet connection and try again.")
        print("You can also manually download the model from:")
        print("https://github.com/coqui-ai/STT-models/releases/") 