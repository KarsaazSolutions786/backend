from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
import hashlib
import secrets
import uuid
import logging

from ..models import User, RefreshToken, LoginAttempt
from ..config import settings

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def hash_token(self, token: str) -> str:
        """Hash a token for storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    async def create_user(self, db: Session, email: str, password: str) -> User:
        """Create a new user"""
        try:
            user_id = str(uuid.uuid4())
            hashed_password = self.hash_password(password)
            
            user = User(
                id=user_id,
                email=email,
                password_hash=hashed_password,
                is_active=True,
                is_verified=False,
                created_at=datetime.utcnow()
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            logger.info(f"User created: {email}")
            return user
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {e}")
            raise
    
    async def authenticate_user(self, db: Session, email: str, password: str) -> User:
        """Authenticate a user by email and password"""
        try:
            user = db.query(User).filter(User.email == email).first()
            
            if not user:
                return None
            
            if not user.is_active:
                return None
            
            if not self.verify_password(password, user.password_hash):
                return None
            
            return user
        
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    async def store_refresh_token(self, db: Session, user_id: str, token: str):
        """Store a refresh token in the database"""
        try:
            token_hash = self.hash_token(token)
            expires_at = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
            
            refresh_token = RefreshToken(
                user_id=user_id,
                token_hash=token_hash,
                expires_at=expires_at
            )
            
            db.add(refresh_token)
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error storing refresh token: {e}")
            raise
    
    async def update_last_login(self, db: Session, user_id: str, ip_address: str = None, user_agent: str = None):
        """Update user's last login timestamp"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
                
                db.commit()
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating last login: {e}")
    
    async def log_login_attempt(self, db: Session, email: str, success: bool, failure_reason: str = None, ip_address: str = None, user_agent: str = None):
        """Log a login attempt"""
        try:
            login_attempt = LoginAttempt(
                email=email,
                ip_address=ip_address or "unknown",
                user_agent=user_agent,
                success=success,
                failure_reason=failure_reason,
                attempted_at=datetime.utcnow()
            )
            
            db.add(login_attempt)
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error logging login attempt: {e}")
    
    async def is_account_locked(self, db: Session, email: str) -> bool:
        """Check if an account is locked due to too many failed attempts"""
        try:
            # Check recent failed attempts
            cutoff_time = datetime.utcnow() - timedelta(minutes=settings.LOCKOUT_DURATION_MINUTES)
            
            failed_attempts = db.query(LoginAttempt).filter(
                LoginAttempt.email == email,
                LoginAttempt.success == False,
                LoginAttempt.attempted_at > cutoff_time
            ).count()
            
            return failed_attempts >= settings.MAX_LOGIN_ATTEMPTS
        
        except Exception as e:
            logger.error(f"Error checking account lock status: {e}")
            return False
    
    async def generate_password_reset_token(self, db: Session, user_id: str) -> str:
        """Generate and store a password reset token"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return None
            
            # Generate secure token
            reset_token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
            
            # Store token
            user.password_reset_token = reset_token
            user.password_reset_expires = expires_at
            
            db.commit()
            
            return reset_token
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error generating password reset token: {e}")
            raise
    
    async def reset_password_with_token(self, db: Session, token: str, new_password: str) -> bool:
        """Reset password using a reset token"""
        try:
            user = db.query(User).filter(
                User.password_reset_token == token,
                User.password_reset_expires > datetime.utcnow()
            ).first()
            
            if not user:
                return False
            
            # Update password
            user.password_hash = self.hash_password(new_password)
            user.password_reset_token = None
            user.password_reset_expires = None
            
            # Revoke all refresh tokens for security
            db.query(RefreshToken).filter(
                RefreshToken.user_id == user.id,
                RefreshToken.is_revoked == False
            ).update({"is_revoked": True})
            
            db.commit()
            
            logger.info(f"Password reset for user: {user.email}")
            return True
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error resetting password: {e}")
            return False
    
    async def update_password(self, db: Session, user_id: str, new_password: str):
        """Update user password"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            
            user.password_hash = self.hash_password(new_password)
            
            # Revoke all refresh tokens for security
            db.query(RefreshToken).filter(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False
            ).update({"is_revoked": True})
            
            db.commit()
            
            logger.info(f"Password updated for user: {user.email}")
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating password: {e}")
            raise
    
    async def deactivate_user(self, db: Session, user_id: str):
        """Deactivate a user account"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                user.is_active = False
                
                # Revoke all refresh tokens
                db.query(RefreshToken).filter(
                    RefreshToken.user_id == user_id,
                    RefreshToken.is_revoked == False
                ).update({"is_revoked": True})
                
                db.commit()
                
                logger.info(f"User deactivated: {user.email}")
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error deactivating user: {e}")
            raise
    
    async def cleanup_expired_tokens(self, db: Session):
        """Clean up expired refresh tokens"""
        try:
            expired_count = db.query(RefreshToken).filter(
                RefreshToken.expires_at < datetime.utcnow()
            ).delete()
            
            db.commit()
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired refresh tokens")
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error cleaning up expired tokens: {e}")
    
    async def get_user_sessions(self, db: Session, user_id: str):
        """Get active sessions for a user"""
        try:
            sessions = db.query(RefreshToken).filter(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False,
                RefreshToken.expires_at > datetime.utcnow()
            ).all()
            
            return [
                {
                    "id": session.id,
                    "created_at": session.created_at,
                    "expires_at": session.expires_at
                }
                for session in sessions
            ]
        
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    async def revoke_session(self, db: Session, user_id: str, session_id: str):
        """Revoke a specific user session"""
        try:
            session = db.query(RefreshToken).filter(
                RefreshToken.id == session_id,
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False
            ).first()
            
            if session:
                session.is_revoked = True
                db.commit()
                
                logger.info(f"Session revoked for user: {user_id}")
                return True
            
            return False
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error revoking session: {e}")
            return False 