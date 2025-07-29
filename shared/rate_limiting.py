"""
Comprehensive Rate Limiting and Brute-Force Protection for Eindr Microservices

This module provides advanced rate limiting, account lockout mechanisms, and
protection against various types of attacks.
"""

import time
import hashlib
import logging
from typing import Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from fastapi import HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
import os
from .redis_manager import get_redis_manager, RedisManager

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting rules"""
    requests: int
    window_seconds: int
    burst_requests: int = 0  # Allow burst up to this many requests
    block_duration_seconds: int = 300  # Block for 5 minutes by default
    
    def __str__(self):
        return f"{self.requests}/{self.window_seconds}s"

class BruteForceProtection:
    """Advanced brute-force protection with exponential backoff"""
    
    def __init__(self, redis_manager: Optional[RedisManager] = None):
        """
        Initialize brute-force protection
        
        Args:
            redis_manager: Optional Redis manager for distributed rate limiting with graceful fallback
        """
        self.redis_manager = redis_manager or get_redis_manager()
        self.local_storage = defaultdict(lambda: {"attempts": 0, "last_attempt": 0, "blocked_until": 0})
        
        # Rate limit configurations for different endpoint types
        self.rate_limits = {
            # Authentication endpoints (most restrictive)
            "auth.login": RateLimitConfig(5, 60, burst_requests=2, block_duration_seconds=300),
            "auth.register": RateLimitConfig(3, 300, block_duration_seconds=600),
            "auth.password_reset": RateLimitConfig(2, 300, block_duration_seconds=900),
            "auth.token_refresh": RateLimitConfig(10, 60),
            
            # API endpoints (moderate restrictions)
            "api.read": RateLimitConfig(100, 60),
            "api.write": RateLimitConfig(50, 60, block_duration_seconds=60),
            "api.delete": RateLimitConfig(20, 60, block_duration_seconds=120),
            "api.upload": RateLimitConfig(10, 300, block_duration_seconds=300),
            
            # AI endpoints (resource intensive)
            "ai.stt": RateLimitConfig(30, 60, block_duration_seconds=120),
            "ai.tts": RateLimitConfig(30, 60, block_duration_seconds=120),
            "ai.chat": RateLimitConfig(20, 60, block_duration_seconds=180),
            "ai.intent": RateLimitConfig(100, 60),
            
            # Search and query endpoints
            "search": RateLimitConfig(50, 60),
            "query": RateLimitConfig(100, 60),
            
            # Default rate limit
            "default": RateLimitConfig(60, 60)
        }
    
    def _get_client_identifier(self, request: Request, customer_id: Optional[int] = None) -> str:
        """
        Get unique identifier for client (IP + Customer ID if available)
        
        Args:
            request: FastAPI request object
            customer_id: Optional authenticated customer ID
            
        Returns:
            Unique client identifier
        """
        ip_address = get_remote_address(request)
        
        # Hash the IP to protect privacy in logs
        ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16]
        
        if customer_id:
            return f"user:{customer_id}:ip:{ip_hash}"
        else:
            return f"ip:{ip_hash}"
    
    def _get_rate_limit_key(self, endpoint: str, client_id: str) -> str:
        """Generate Redis key for rate limiting"""
        return f"rate_limit:{endpoint}:{client_id}"
    
    def _get_block_key(self, endpoint: str, client_id: str) -> str:
        """Generate Redis key for blocking"""
        return f"block:{endpoint}:{client_id}"
    
    def _check_rate_limit_redis(self, key: str, config: RateLimitConfig) -> Tuple[bool, int, int]:
        """
        Check rate limit using Redis sliding window with graceful fallback
        
        Args:
            key: Redis key for this rate limit
            config: Rate limit configuration
            
        Returns:
            Tuple of (allowed, current_count, reset_time)
        """
        if not self.redis_manager.is_available:
            return self._check_rate_limit_local(key, config)
        
        with self.redis_manager.safe_operation("rate_limit_check") as client:
            if not client:
                return self._check_rate_limit_local(key, config)
            
            try:
                pipe = client.pipeline()
                now = time.time()
                window_start = now - config.window_seconds
                
                # Remove old entries
                pipe.zremrangebyscore(key, 0, window_start)
                
                # Count current requests
                pipe.zcard(key)
                
                # Add current request
                pipe.zadd(key, {str(now): now})
                
                # Set expiration
                pipe.expire(key, config.window_seconds)
                
                results = self.redis_manager.execute_pipeline(pipe, "rate_limit_check")
                if not results:
                    return self._check_rate_limit_local(key, config)
                
                current_count = results[1] + 1  # +1 for the request we just added
                
                # Check if we're within limits
                limit = config.burst_requests if config.burst_requests > 0 else config.requests
                allowed = current_count <= limit
                
                reset_time = int(now + config.window_seconds)
                
                return allowed, current_count, reset_time
                
            except Exception as e:
                logger.warning(f"Redis rate limiting error, falling back to local: {e}")
                return self._check_rate_limit_local(key, config)
    
    def _check_rate_limit_local(self, key: str, config: RateLimitConfig) -> Tuple[bool, int, int]:
        """
        Check rate limit using local memory (fallback)
        
        Args:
            key: Local storage key
            config: Rate limit configuration
            
        Returns:
            Tuple of (allowed, current_count, reset_time)
        """
        now = time.time()
        storage = self.local_storage[key]
        
        # Initialize if first request
        if "requests" not in storage:
            storage["requests"] = deque()
        
        # Remove old requests outside the window
        window_start = now - config.window_seconds
        while storage["requests"] and storage["requests"][0] < window_start:
            storage["requests"].popleft()
        
        # Add current request
        storage["requests"].append(now)
        
        current_count = len(storage["requests"])
        limit = config.burst_requests if config.burst_requests > 0 else config.requests
        allowed = current_count <= limit
        
        reset_time = int(now + config.window_seconds)
        
        return allowed, current_count, reset_time
    
    def check_rate_limit(self, request: Request, endpoint: str, customer_id: Optional[int] = None) -> Dict:
        """
        Check if request is within rate limits
        
        Args:
            request: FastAPI request object
            endpoint: Endpoint identifier (e.g., "auth.login", "api.read")
            customer_id: Optional authenticated customer ID
            
        Returns:
            Dictionary with rate limit status
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        client_id = self._get_client_identifier(request, customer_id)
        config = self.rate_limits.get(endpoint, self.rate_limits["default"])
        
        # Check if client is currently blocked
        if self._is_blocked(endpoint, client_id):
            block_info = self._get_block_info(endpoint, client_id)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "blocked_until": block_info["blocked_until"],
                    "retry_after": block_info["retry_after"]
                },
                headers={"Retry-After": str(block_info["retry_after"])}
            )
        
        # Check rate limit
        rate_limit_key = self._get_rate_limit_key(endpoint, client_id)
        
        if self.redis_manager.is_available:
            allowed, current_count, reset_time = self._check_rate_limit_redis(rate_limit_key, config)
        else:
            allowed, current_count, reset_time = self._check_rate_limit_local(rate_limit_key, config)
        
        if not allowed:
            # Block the client if they've exceeded the limit
            self._block_client(endpoint, client_id, config.block_duration_seconds)
            
            logger.warning(f"Rate limit exceeded for {client_id} on {endpoint}: {current_count}/{config.requests}")
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": config.requests,
                    "window": config.window_seconds,
                    "current": current_count,
                    "reset_time": reset_time,
                    "blocked_duration": config.block_duration_seconds
                },
                headers={"Retry-After": str(config.block_duration_seconds)}
            )
        
        return {
            "allowed": True,
            "limit": config.requests,
            "remaining": max(0, config.requests - current_count),
            "reset_time": reset_time
        }
    
    def _is_blocked(self, endpoint: str, client_id: str) -> bool:
        """Check if client is currently blocked"""
        block_key = self._get_block_key(endpoint, client_id)
        
        # Try Redis first with graceful fallback
        blocked_until = self.redis_manager.get(block_key)
        if blocked_until:
            try:
                return time.time() < float(blocked_until)
            except (ValueError, TypeError):
                pass
        
        # Fallback to local storage
        storage = self.local_storage[block_key]
        return time.time() < storage.get("blocked_until", 0)
    
    def _get_block_info(self, endpoint: str, client_id: str) -> Dict:
        """Get information about current block"""
        block_key = self._get_block_key(endpoint, client_id)
        
        # Try Redis first with graceful fallback
        blocked_until = self.redis_manager.get(block_key)
        if blocked_until:
            try:
                blocked_until_time = float(blocked_until)
                return {
                    "blocked_until": int(blocked_until_time),
                    "retry_after": int(max(0, blocked_until_time - time.time()))
                }
            except (ValueError, TypeError):
                pass
        
        # Fallback to local storage
        storage = self.local_storage[block_key]
        blocked_until = storage.get("blocked_until", 0)
        return {
            "blocked_until": int(blocked_until),
            "retry_after": int(max(0, blocked_until - time.time()))
        }
    
    def _block_client(self, endpoint: str, client_id: str, duration_seconds: int):
        """Block client for specified duration"""
        block_key = self._get_block_key(endpoint, client_id)
        blocked_until = time.time() + duration_seconds
        
        # Try Redis with graceful fallback
        self.redis_manager.set(block_key, str(blocked_until), ex=duration_seconds)
        
        # Also store in local storage as fallback
        self.local_storage[block_key]["blocked_until"] = blocked_until
    
    def record_failed_attempt(self, request: Request, endpoint: str, customer_id: Optional[int] = None):
        """
        Record a failed authentication attempt
        
        Args:
            request: FastAPI request object
            endpoint: Endpoint identifier
            customer_id: Optional customer ID if known
        """
        client_id = self._get_client_identifier(request, customer_id)
        config = self.rate_limits.get(endpoint, self.rate_limits["default"])
        
        # Increase the severity for failed authentication attempts
        if endpoint.startswith("auth."):
            # Progressive penalty: first failure = 1 request, second = 2 requests, etc.
            penalty_key = f"penalty:{endpoint}:{client_id}"
            
            # Try Redis with graceful fallback
            penalty_count = self.redis_manager.incr(penalty_key)
            if penalty_count:
                self.redis_manager.expire(penalty_key, config.window_seconds * 2)
            else:
                # Fallback to local storage
                storage = self.local_storage[penalty_key]
                penalty_count = storage.get("count", 0) + 1
                storage["count"] = penalty_count
                storage["expires"] = time.time() + (config.window_seconds * 2)
            
            # Apply exponential backoff for repeated failures
            additional_block_time = min(penalty_count * 60, 3600)  # Max 1 hour
            self._block_client(endpoint, client_id, config.block_duration_seconds + additional_block_time)
            
            logger.warning(f"Failed attempt #{penalty_count} for {client_id} on {endpoint}")

# Global rate limiting instance with enhanced Redis manager
brute_force_protection = BruteForceProtection()

# SlowAPI limiter for basic rate limiting
limiter = Limiter(key_func=get_remote_address)

def rate_limit(endpoint: str):
    """
    Decorator for rate limiting endpoints
    
    Args:
        endpoint: Endpoint identifier for rate limit configuration
    """
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            # Extract customer ID if available
            customer_id = None
            if hasattr(request.state, "customer_id"):
                customer_id = request.state.customer_id
            
            # Check rate limit
            brute_force_protection.check_rate_limit(request, endpoint, customer_id)
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator

def record_auth_failure(request: Request, endpoint: str = "auth.login", customer_id: Optional[int] = None):
    """
    Record an authentication failure for brute-force protection
    
    Args:
        request: FastAPI request object
        endpoint: Authentication endpoint identifier
        customer_id: Optional customer ID if known
    """
    brute_force_protection.record_failed_attempt(request, endpoint, customer_id)

# Rate limiting middleware for FastAPI
class RateLimitMiddleware:
    """Middleware to apply rate limiting to all requests"""
    
    def __init__(self, app, default_endpoint: str = "default"):
        self.app = app
        self.default_endpoint = default_endpoint
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Determine endpoint type based on path
            path = request.url.path
            endpoint = self._get_endpoint_type(path)
            
            try:
                # Check rate limit
                rate_info = brute_force_protection.check_rate_limit(request, endpoint)
                
                # Add rate limit headers to response
                async def send_wrapper(message):
                    if message["type"] == "http.response.start":
                        headers = dict(message.get("headers", []))
                        headers[b"X-RateLimit-Limit"] = str(rate_info["limit"]).encode()
                        headers[b"X-RateLimit-Remaining"] = str(rate_info["remaining"]).encode()
                        headers[b"X-RateLimit-Reset"] = str(rate_info["reset_time"]).encode()
                        message["headers"] = list(headers.items())
                    await send(message)
                
                await self.app(scope, receive, send_wrapper)
                
            except HTTPException as e:
                # Convert HTTPException to ASGI response
                response_data = {
                    "type": "http.response.start",
                    "status": e.status_code,
                    "headers": [(b"content-type", b"application/json")]
                }
                
                if "Retry-After" in e.headers:
                    response_data["headers"].append((b"retry-after", e.headers["Retry-After"].encode()))
                
                await send(response_data)
                await send({
                    "type": "http.response.body",
                    "body": str(e.detail).encode()
                })
        else:
            await self.app(scope, receive, send)
    
    def _get_endpoint_type(self, path: str) -> str:
        """Determine endpoint type from request path"""
        if "/auth/" in path or path.endswith("/login") or path.endswith("/register"):
            if "login" in path:
                return "auth.login"
            elif "register" in path:
                return "auth.register" 
            elif "password" in path:
                return "auth.password_reset"
            elif "refresh" in path:
                return "auth.token_refresh"
            else:
                return "auth.general"
        elif "/search" in path:
            return "search"
        elif any(ai_path in path for ai_path in ["/stt", "/tts", "/chat", "/intent"]):
            if "/stt" in path:
                return "ai.stt"
            elif "/tts" in path:
                return "ai.tts"
            elif "/chat" in path:
                return "ai.chat"
            elif "/intent" in path:
                return "ai.intent"
        elif any(method in path for method in ["DELETE", "delete"]):
            return "api.delete"
        elif any(method in path for method in ["upload", "file"]):
            return "api.upload"
        elif any(method in path for method in ["POST", "PUT", "PATCH"]):
            return "api.write"
        else:
            return "api.read"