# Global service instances (will be set by main.py)
stt_service = None
tts_service = None
intent_service = None
chat_service = None

def get_stt_service():
    return stt_service

def get_tts_service():
    return tts_service

def get_intent_service():
    return intent_service

def get_chat_service():
    return chat_service

def set_services(stt, tts, intent, chat):
    """Set the global service instances."""
    global stt_service, tts_service, intent_service, chat_service
    stt_service = stt
    tts_service = tts
    intent_service = intent
    chat_service = chat 