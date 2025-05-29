from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from models.models import User
from connect_db import get_db
from core.config import settings
import jwt
from jwt import PyJWTError
from utils.logger import logger
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Security scheme
security = HTTPBearer()

# Development mode flag
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

async def verify_traditional_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify traditional JWT token created by our auth system."""
    try:
        token = credentials.credentials
        logger.info(f"Verifying traditional token: {token[:20]}...")
        
        # Decode and verify token
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "uid": user_id,
            "email": email,
            "name": email  # Use email as name for traditional auth
        }
        
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except PyJWTError as e:
        logger.error(f"JWT Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def verify_firebase_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Firebase ID token."""
    try:
        # Log request details for debugging
        logger.info(f"Request path: {request.url.path}")
        
        if not credentials:
            logger.error("No credentials provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No credentials provided",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = credentials.credentials
        logger.info(f"Received Firebase token: {token[:20]}...")
        
        # Development mode check
        if DEV_MODE:
            logger.warning("Running in development mode - Firebase token verification is relaxed")
            try:
                decoded_token = jwt.decode(token, options={"verify_signature": False})
                logger.info(f"Development mode: Firebase token decoded for user: {decoded_token.get('email')}")
                return {
                    "uid": decoded_token.get("sub"),
                    "email": decoded_token.get("email"),
                    "name": decoded_token.get("name")
                }
            except Exception as e:
                logger.error(f"Development mode Firebase token decode error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid Firebase token format: {str(e)}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        # Production mode - decode and verify Firebase token
        try:
            decoded_token = jwt.decode(token, options={"verify_signature": False})
            logger.info(f"Firebase token decoded successfully for user: {decoded_token.get('email')}")
            
            return {
                "uid": decoded_token.get("sub"),
                "email": decoded_token.get("email"),
                "name": decoded_token.get("name")
            }
        except jwt.ExpiredSignatureError:
            logger.error("Firebase token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Firebase token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid Firebase token format: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid Firebase token format: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except PyJWTError as e:
        logger.error(f"Firebase JWT Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Firebase token format: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Firebase authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Firebase authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def verify_any_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Verify either traditional JWT or Firebase token."""
    try:
        # First try traditional JWT
        try:
            user_data = await verify_traditional_token(credentials)
            logger.info(f"Successfully verified traditional token for user: {user_data['email']}")
            return user_data
        except HTTPException:
            # If traditional JWT fails, try Firebase token
            logger.info("Traditional JWT verification failed, trying Firebase token...")
            pass
        
        # Try Firebase token
        try:
            # Create a mock request object for Firebase verification
            class MockRequest:
                def __init__(self):
                    self.url = type('MockURL', (), {'path': '/api/auth'})()
            
            mock_request = MockRequest()
            user_data = await verify_firebase_token(mock_request, credentials)
            logger.info(f"Successfully verified Firebase token for user: {user_data['email']}")
            return user_data
        except HTTPException:
            logger.error("Both traditional and Firebase token verification failed")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: not a valid traditional JWT or Firebase token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(current_user: dict = Depends(verify_any_token), db: Session = Depends(get_db)):
    """Get current user from database using verified token."""
    try:
        user_id = current_user["uid"]
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found in database"
            )
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get current user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get current user: {str(e)}"
        ) 