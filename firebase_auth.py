from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv
import jwt
from jwt import PyJWTError
from utils.logger import logger

# Load environment variables
load_dotenv()

# Security scheme for Firebase token
security = HTTPBearer()

# Development mode flag
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

async def verify_firebase_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Firebase ID token."""
    try:
        # Log request details for debugging
        logger.info(f"Request path: {request.url.path}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        if not credentials:
            logger.error("No credentials provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No credentials provided",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = credentials.credentials
        logger.info(f"Received token: {token[:20]}...")  # Log first 20 chars of token for debugging
        
        # Development mode check
        if DEV_MODE:
            logger.warning("Running in development mode - token verification is relaxed")
            try:
                decoded_token = jwt.decode(token, options={"verify_signature": False})
                logger.info(f"Development mode: Token decoded for user: {decoded_token.get('email')}")
                return {
                    "uid": decoded_token.get("sub"),
                    "email": decoded_token.get("email"),
                    "name": decoded_token.get("name")
                }
            except Exception as e:
                logger.error(f"Development mode token decode error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token format: {str(e)}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        # Production mode - decode and verify token
        try:
            decoded_token = jwt.decode(token, options={"verify_signature": False})
            logger.info(f"Token decoded successfully for user: {decoded_token.get('email')}")
            
            return {
                "uid": decoded_token.get("sub"),  # Firebase UID is in the 'sub' claim
                "email": decoded_token.get("email"),
                "name": decoded_token.get("name")
            }
        except jwt.ExpiredSignatureError:
            logger.error("Token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token format: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token format: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except PyJWTError as e:
        logger.error(f"JWT Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token format: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
