"""
Secure Refresh Token Service for Eindr Microservices

This module provides secure refresh token handling with proper storage,
rotation, and revocation mechanisms.
"""

import secrets
import hashlib
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
from fastapi import HTTPException, status
import json
import os

logger = logging.getLogger(__name__)

# Base for refresh token models
RefreshTokenBase = declarative_base()

class RefreshToken(RefreshTokenBase):
    """Refresh token model for secure token storage"""
    __tablename__ = "refresh_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256 hash
    device_id = Column(String(255))  # Device identifier
    user_agent = Column(Text)
    ip_address = Column(String(45))  # IPv6 compatible
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, default=datetime.utcnow)
    is_revoked = Column(Boolean, default=False, index=True)
    revoked_at = Column(DateTime)
    revoked_reason = Column(String(100))  # 'logout', 'security', 'expired', 'rotation'
    rotation_count = Column(Integer, default=0)  # Track token rotations
    parent_token_hash = Column(String(64))  # For token family tracking
    
    def is_valid(self) -> bool:
        """Check if token is still valid"""
        return (
            not self.is_revoked and
            self.expires_at > datetime.utcnow()
        )
    
    def revoke(self, reason: str = "manual"):
        """Revoke the token"""
        self.is_revoked = True
        self.revoked_at = datetime.utcnow()
        self.revoked_reason = reason

class RefreshTokenService:
    """Service for managing refresh tokens securely"""
    
    def __init__(self, db: Session):
        self.db = db
        
        # Configuration
        self.token_length = int(os.getenv("REFRESH_TOKEN_LENGTH", "32"))
        self.token_expiry_days = int(os.getenv("REFRESH_TOKEN_EXPIRY_DAYS", "30"))
        self.max_tokens_per_user = int(os.getenv("MAX_REFRESH_TOKENS_PER_USER", "5"))
        self.enable_token_rotation = os.getenv("ENABLE_TOKEN_ROTATION", "true").lower() == "true"
        self.revoke_family_on_reuse = os.getenv("REVOKE_FAMILY_ON_REUSE", "true").lower() == "true"
        
        logger.info("RefreshTokenService initialized with database-only mode")
    
    def generate_token(self) -> str:
        """Generate a cryptographically secure refresh token"""
        return secrets.token_urlsafe(self.token_length)
    
    def hash_token(self, token: str) -> str:
        """Hash token for secure storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def create_refresh_token(
        self,
        customer_id: int,
        device_id: str = None,
        user_agent: str = None,
        ip_address: str = None,
        parent_token_hash: str = None
    ) -> Tuple[str, RefreshToken]:
        """
        Create a new refresh token
        
        Args:
            customer_id: Customer ID
            device_id: Device identifier
            user_agent: User agent string
            ip_address: Client IP address
            parent_token_hash: Hash of parent token (for rotation)
            
        Returns:
            Tuple of (plain_token, token_record)
        """
        try:
            # Clean up expired tokens first
            self._cleanup_expired_tokens(customer_id)
            
            # Check token limit per user
            active_tokens = self.db.query(RefreshToken).filter(
                RefreshToken.customer_id == customer_id,
                RefreshToken.is_revoked == False,
                RefreshToken.expires_at > datetime.utcnow()
            ).count()
            
            if active_tokens >= self.max_tokens_per_user:
                # Revoke oldest token
                oldest_token = self.db.query(RefreshToken).filter(
                    RefreshToken.customer_id == customer_id,
                    RefreshToken.is_revoked == False
                ).order_by(RefreshToken.last_used_at.asc()).first()
                
                if oldest_token:
                    oldest_token.revoke("limit_exceeded")
            
            # Generate new token
            plain_token = self.generate_token()
            token_hash = self.hash_token(plain_token)
            
            # Calculate expiry
            expires_at = datetime.utcnow() + timedelta(days=self.token_expiry_days)
            
            # Create token record
            token_record = RefreshToken(
                customer_id=customer_id,
                token_hash=token_hash,
                device_id=device_id,
                user_agent=user_agent,
                ip_address=ip_address,
                expires_at=expires_at,
                parent_token_hash=parent_token_hash
            )
            
            # If this is a rotated token, increment rotation count
            if parent_token_hash:
                parent_token = self.db.query(RefreshToken).filter(
                    RefreshToken.token_hash == parent_token_hash
                ).first()
                if parent_token:
                    token_record.rotation_count = parent_token.rotation_count + 1
            
            self.db.add(token_record)
            self.db.commit()
            self.db.refresh(token_record)
            
            # Cache in Redis if available
            if self.redis_client:
                try:
                    self._cache_token(token_hash, token_record)
                    logger.debug(f"Created and cached new refresh token for customer {customer_id}")
                except Exception as e:
                    logger.error(f"Failed to cache new token: {e}")
                    logger.warning("Token created but not cached - will use database for validation")
                    # Continue execution - the token is still created in the database
            else:
                logger.info(f"Created refresh token for customer {customer_id}")
            
            return plain_token, token_record
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            # Rollback the transaction if an error occurs
            self.db.rollback()
            raise  # Re-raise the exception to be handled by the caller
    
    def validate_token(self, token: str) -> Optional[RefreshToken]:
        """
        Validate a refresh token
        
        Args:
            token: Plain text token
            
        Returns:
            RefreshToken record if valid, None otherwise
        """
        token_hash = self.hash_token(token)
        
        # Check database
        token_record = self.db.query(RefreshToken).filter(
            RefreshToken.token_hash == token_hash
        ).first()
        
        if not token_record:
            return None
        
        # Check if token is valid
        if not token_record.is_valid():
            return None
        
        # Update last used timestamp
        token_record.last_used_at = datetime.utcnow()
        self.db.commit()
        
        # Update cache
        if self.redis_client:
            try:
                self._cache_token(token_hash, token_record)
            except Exception as e:
                logger.error(f"Failed to update token in cache: {e}")
                # Continue execution - the token is still valid
        
        return token_record
    
    def rotate_token(
        self,
        old_token: str,
        device_id: str = None,
        user_agent: str = None,
        ip_address: str = None
    ) -> Optional[Tuple[str, RefreshToken]]:
        """
        Rotate a refresh token (replace with new one)
        
        Args:
            old_token: Current token to rotate
            device_id: Device identifier
            user_agent: User agent string
            ip_address: Client IP address
            
        Returns:
            Tuple of (new_token, token_record) or None if failed
        """
        if not self.enable_token_rotation:
            return None
        
        # Validate old token
        old_token_record = self.validate_token(old_token)
        if not old_token_record:
            return None
        
        # Check for token reuse (security concern)
        if old_token_record.is_revoked:
            if self.revoke_family_on_reuse:
                self._revoke_token_family(old_token_record.parent_token_hash or old_token_record.token_hash)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token reuse detected"
            )
        
        # Revoke old token
        old_token_record.revoke("rotation")
        
        # Create new token
        new_token, new_token_record = self.create_refresh_token(
            customer_id=old_token_record.customer_id,
            device_id=device_id or old_token_record.device_id,
            user_agent=user_agent or old_token_record.user_agent,
            ip_address=ip_address or old_token_record.ip_address,
            parent_token_hash=old_token_record.parent_token_hash or old_token_record.token_hash
        )
        
        # Remove old token from cache
        if self.redis_client:
            self._remove_cached_token(old_token_record.token_hash)
        
        logger.info(f"Rotated refresh token for customer {old_token_record.customer_id}")
        return new_token, new_token_record
    
    def is_token_revoked(self, token: str) -> bool:
        """
        Check if a token has been revoked
        
        Args:
            token: Token to check
            
        Returns:
            True if token is revoked or invalid
        """
        token_hash = self.hash_token(token)
        
        # Check database
        token_record = self.db.query(RefreshToken).filter(
            RefreshToken.token_hash == token_hash
        ).first()
        
        if not token_record:
            return True  # Token doesn't exist, consider it revoked
        
        return token_record.is_revoked or token_record.expires_at <= datetime.utcnow()
    
    def revoke_token(self, token: str, reason: str = "manual") -> bool:
        """
        Revoke a specific refresh token
        
        Args:
            token: Token to revoke
            reason: Reason for revocation
            
        Returns:
            True if successful
        """
        try:
            token_hash = self.hash_token(token)
            
            token_record = self.db.query(RefreshToken).filter(
                RefreshToken.token_hash == token_hash
            ).first()
            
            if not token_record:
                logger.warning(f"Attempted to revoke non-existent token: {token_hash[:10]}...")
                return False
            
            token_record.revoke(reason)
            self.db.commit()
            
            logger.info(f"Revoked refresh token for customer {token_record.customer_id}, reason: {reason}")
            return True
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False
    
    def revoke_all_tokens(self, customer_id: int, reason: str = "logout_all") -> int:
        """
        Revoke all tokens for a customer
        
        Args:
            customer_id: Customer ID
            reason: Reason for revocation
            
        Returns:
            Number of tokens revoked
        """
        try:
            tokens = self.db.query(RefreshToken).filter(
                RefreshToken.customer_id == customer_id,
                RefreshToken.is_revoked == False
            ).all()
            
            count = 0
            
            for token in tokens:
                token.revoke(reason)
                count += 1
            
            self.db.commit()
            
            logger.info(f"Revoked {count} refresh tokens for customer {customer_id}, reason: {reason}")
            
            return count
        except Exception as e:
            logger.error(f"Bulk token revocation failed for customer {customer_id}: {e}")
            return 0
    
    def revoke_device_tokens(self, customer_id: int, device_id: str, reason: str = "device_logout") -> int:
        """
        Revoke all tokens for a specific device
        
        Args:
            customer_id: Customer ID
            device_id: Device identifier
            reason: Reason for revocation
            
        Returns:
            Number of tokens revoked
        """
        try:
            tokens = self.db.query(RefreshToken).filter(
                RefreshToken.customer_id == customer_id,
                RefreshToken.device_id == device_id,
                RefreshToken.is_revoked == False
            ).all()
            
            count = 0
            
            for token in tokens:
                token.revoke(reason)
                count += 1
            
            self.db.commit()
            
            logger.info(f"Revoked {count} refresh tokens for customer {customer_id} device {device_id}, reason: {reason}")
            
            return count
        except Exception as e:
            logger.error(f"Device token revocation failed for customer {customer_id}, device {device_id}: {e}")
            return 0
    
    def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired and revoked tokens
        
        Returns:
            Number of tokens cleaned up
        """
        try:
            # Delete tokens expired more than 7 days ago
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            expired_tokens = self.db.query(RefreshToken).filter(
                RefreshToken.expires_at < cutoff_date
            ).all()
            
            count = 0
            
            for token in expired_tokens:
                self.db.delete(token)
                count += 1
            
            self.db.commit()
            
            logger.info(f"Cleaned up {count} expired refresh tokens")
            
            return count
        except Exception as e:
            logger.error(f"Token cleanup failed: {e}")
            return 0
    
    def get_active_sessions(self, customer_id: int) -> List[Dict]:
        """
        Get active sessions for a customer
        
        Args:
            customer_id: Customer ID
            
        Returns:
            List of active session information
        """
        tokens = self.db.query(RefreshToken).filter(
            RefreshToken.customer_id == customer_id,
            RefreshToken.is_revoked == False,
            RefreshToken.expires_at > datetime.utcnow()
        ).order_by(RefreshToken.last_used_at.desc()).all()
        
        sessions = []
        for token in tokens:
            sessions.append({
                "device_id": token.device_id,
                "user_agent": token.user_agent,
                "ip_address": token.ip_address,
                "created_at": token.created_at.isoformat(),
                "last_used_at": token.last_used_at.isoformat(),
                "expires_at": token.expires_at.isoformat()
            })
        
        return sessions
    
    def _cleanup_expired_tokens(self, customer_id: int):
        """Clean up expired tokens for a specific customer"""
        try:
            # Find expired tokens
            expired_tokens = self.db.query(RefreshToken).filter(
                RefreshToken.customer_id == customer_id,
                RefreshToken.expires_at < datetime.utcnow(),
                RefreshToken.is_revoked == False
            ).all()
            
            count = 0
            
            for token in expired_tokens:
                token.revoke("expired")
                count += 1
            
            if count > 0:
                self.db.commit()
                logger.info(f"Cleaned up {count} expired tokens for customer {customer_id}")
        except Exception as e:
            logger.error(f"Failed to clean up expired tokens for customer {customer_id}: {e}")
            # Continue execution - we'll try again later
    
    def _revoke_token_family(self, family_hash: str):
        """Revoke all tokens in a family (for security)"""
        try:
            family_tokens = self.db.query(RefreshToken).filter(
                (RefreshToken.token_hash == family_hash) |
                (RefreshToken.parent_token_hash == family_hash)
            ).all()
            
            revoked_count = 0
            
            for token in family_tokens:
                if not token.is_revoked:
                    token.revoke("family_revocation")
                    revoked_count += 1
            
            if revoked_count > 0:
                self.db.commit()
                logger.warning(f"Revoked {revoked_count} tokens in family {family_hash[:10]}... due to security concern")
        except Exception as e:
            logger.error(f"Failed to revoke token family {family_hash[:10]}...: {e}")
            # Continue execution - we'll try to revoke as many as possible
    
def create_refresh_token_service(db: Session) -> RefreshTokenService:
    """Create a refresh token service instance"""
    return RefreshTokenService(db)