from typing import Optional, Dict
import logging
import json
import base64

logger = logging.getLogger(__name__)

class AuthService:
    """Stub implementation of auth service for reminder service"""
    
    async def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token and return customer info"""
        try:
            # For development, decode the JWT token without signature verification
            # In production, this should call the actual auth service
            
            if not token:
                return None
            
            # Manual JWT decode (payload is the second part)
            parts = token.split('.')
            if len(parts) != 3:
                logger.warning("Invalid JWT token format")
                return None
            
            payload_part = parts[1]
            # Add padding if needed
            payload_part += '=' * (4 - len(payload_part) % 4)
            
            try:
                decoded_bytes = base64.b64decode(payload_part)
                payload = json.loads(decoded_bytes)
                
                # Extract customer ID from 'sub' field
                customer_id = payload.get('sub')
                if customer_id:
                    # Return format that matches what other services expect
                    return {
                        "user_id": customer_id,  # This should be the string customer ID from JWT
                        "customer_id": int(customer_id),  # Also provide as int for convenience
                        "email": f"user{customer_id}@eindr.com"  # Placeholder
                    }
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to decode JWT payload: {e}")
                return None
            
            logger.warning("No customer ID found in token")
            return None
            
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return None 