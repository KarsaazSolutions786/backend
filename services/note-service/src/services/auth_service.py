import jwt
import json
import base64
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def get_current_customer(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Extract customer information from JWT token.
    This is a simplified version that decodes the JWT without signature verification
    for development purposes.
    """
    try:
        token = credentials.credentials
        
        # Split the JWT token
        parts = token.split('.')
        if len(parts) != 3:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format"
            )
        
        # Decode the payload (second part)
        payload_part = parts[1]
        # Add padding if needed
        payload_part += '=' * (4 - len(payload_part) % 4)
        
        # Decode base64
        payload_bytes = base64.urlsafe_b64decode(payload_part)
        payload = json.loads(payload_bytes.decode('utf-8'))
        
        # Extract customer ID from 'sub' field
        customer_id = payload.get('sub')
        if not customer_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject"
            )
        
        # Convert to integer if it's a string
        try:
            customer_id_int = int(customer_id)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid customer ID in token"
            )
        
        return {
            "id": customer_id_int,
            "user_id": str(customer_id),  # For backward compatibility
            "customer_id": customer_id_int
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}"
        ) 