#!/usr/bin/env python3
"""Simple gTTS test script"""
import asyncio
from services.tts_service import TextToSpeechService

async def test_gtts():
    tts = TextToSpeechService()
    # Force gTTS usage
    tts.preferred_engine = 'gtts'
    audio = await tts.synthesize_speech('Hello! This is a gTTS test.')
    if audio:
        with open('test_gtts_output.mp3', 'wb') as f:
            f.write(audio)
        print('✅ gTTS test successful!')
        print(f'Audio file size: {len(audio)} bytes')
    else:
        print('❌ gTTS test failed')

if __name__ == "__main__":
    asyncio.run(test_gtts()) 