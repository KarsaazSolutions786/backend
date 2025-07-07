import logging
import asyncio

logger = logging.getLogger(__name__)

class EventPublisher:
    """Placeholder event publisher that logs instead of sending to a message bus."""
    async def publish_event(self, event_name: str, payload: dict):
        logger.info(f"[Stub] Event published: {event_name} -> {payload}")
        # Mimic async I/O
        await asyncio.sleep(0)
        return True 