from services.intent_service import IntentService
# Global service instances (will be set by main.py)
stt_service = None
tts_service = None
# intent_service = None # Removed global intent_service
chat_service = None
intent_processor_service = None # Added global intent_processor_service

_intent_services_cache: dict[str, IntentService] = {} # Added cache for IntentService instances

def get_stt_service():
    # TODO: Consider if STTService needs to be language-aware at instantiation
    # If so, this might need a language_code parameter and caching similar to IntentService.
    return stt_service

def get_tts_service():
    # TODO: Consider if TTSService needs to be language-aware at instantiation
    # (e.g., to load language-specific voice models by default).
    # If so, this might need a language_code parameter and caching.
    return tts_service

def get_intent_service(language_code: str = "en") -> IntentService: # Modified signature
    """
    Returns an IntentService instance for the given language code.
    Instances are cached to avoid reinitialization.
    """
    if language_code not in _intent_services_cache:
        # Assuming IntentService can be initialized with just language_code.
        # If it requires other dependencies, this model might need adjustment,
        # potentially by having main.py pass a factory or shared configurations.
        _intent_services_cache[language_code] = IntentService(language_code=language_code)
        # Consider adding logging for new instance creation if desired.
        # from utils.logger import logger
        # logger.info(f"Created new IntentService instance for language: {language_code}")
    return _intent_services_cache[language_code]

def get_chat_service():
    # TODO: Consider if ChatService needs to be language-aware at instantiation.
    # If so, this might need a language_code parameter and caching.
    return chat_service

def get_intent_processor_service(): # Added getter for intent_processor_service
    """Returns the global IntentProcessorService instance."""
    # This service handles language_code per method call, so one instance is likely fine.
    return intent_processor_service

def set_services(stt, tts, chat, processor): # Modified signature
    """Set the global service instances."""
    global stt_service, tts_service, chat_service, intent_processor_service # Removed intent_service, added intent_processor_service
    stt_service = stt
    tts_service = tts
    chat_service = chat
    intent_processor_service = processor # Set intent_processor_service 