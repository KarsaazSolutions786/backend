"""Enhanced Redis Connection Manager with Graceful Fallback

This module provides a robust Redis connection manager that handles connection failures
gracefully and implements circuit breaker patterns for better resilience.
"""

import time
import logging
from typing import Optional, Any, Dict, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import redis
import os
import json
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open" # Testing if service is back

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Number of failures before opening
    recovery_timeout: int = 60          # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close circuit
    timeout: int = 5                    # Redis operation timeout

class RedisCircuitBreaker:
    """Circuit breaker for Redis operations"""
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if we should try half-open
            if (self.last_failure_time and 
                time.time() - self.last_failure_time > self.config.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Redis circuit breaker: Transitioning to HALF_OPEN")
                return True
            return False
        
        # HALF_OPEN state
        return True
    
    def record_success(self):
        """Record successful operation"""
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Redis circuit breaker: Transitioning to CLOSED")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    def record_failure(self):
        """Record failed operation"""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if (self.state == CircuitState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self.state = CircuitState.OPEN
            logger.warning(f"Redis circuit breaker: Transitioning to OPEN after {self.failure_count} failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Redis circuit breaker: Transitioning back to OPEN from HALF_OPEN")

class RedisManager:
    """Enhanced Redis connection manager with graceful fallback"""
    
    def __init__(self, redis_url: str = None, circuit_breaker_config: CircuitBreakerConfig = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.client: Optional[redis.Redis] = None
        self.circuit_breaker = RedisCircuitBreaker(circuit_breaker_config)
        self.connection_pool = None
        self.is_available = False
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Redis connection with error handling"""
        if not self.redis_url:
            logger.info("No Redis URL provided, Redis features will be disabled")
            return
        
        try:
            # Create connection pool for better performance <mcreference link="https://redis.io/learn/develop/python/fastapi" index="1">1</mcreference>
            self.connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            self.client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            self.client.ping()
            self.is_available = True
            self.circuit_breaker.record_success()
            logger.info("Redis connection established successfully")
            
        except ImportError:
            logger.warning("Redis library not available, Redis features will be disabled")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.circuit_breaker.record_failure()
            self.is_available = False
    
    def _health_check(self) -> bool:
        """Perform periodic health check"""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return self.is_available
        
        self.last_health_check = current_time
        
        if not self.client:
            return False
        
        try:
            self.client.ping()
            if not self.is_available:
                logger.info("Redis connection restored")
            self.is_available = True
            self.circuit_breaker.record_success()
            return True
        except Exception as e:
            if self.is_available:
                logger.warning(f"Redis connection lost: {e}")
            self.is_available = False
            self.circuit_breaker.record_failure()
            return False
    
    @contextmanager
    def safe_operation(self, operation_name: str = "redis_operation"):
        """Context manager for safe Redis operations with circuit breaker <mcreference link="https://www.michaco.net/blog/WhatIfRedisStopsWorkingHowDoIkeepMyAppRunning" index="3">3</mcreference>"""
        if not self.circuit_breaker.can_execute():
            logger.debug(f"Redis circuit breaker is OPEN, skipping {operation_name}")
            yield None
            return
        
        if not self._health_check():
            logger.debug(f"Redis health check failed, skipping {operation_name}")
            self.circuit_breaker.record_failure()
            yield None
            return
        
        try:
            yield self.client
            self.circuit_breaker.record_success()
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error in {operation_name}: {e}")
            self.circuit_breaker.record_failure()
            self.is_available = False
        except redis.TimeoutError as e:
            logger.warning(f"Redis timeout error in {operation_name}: {e}")
            self.circuit_breaker.record_failure()
        except Exception as e:
            logger.error(f"Unexpected Redis error in {operation_name}: {e}")
            self.circuit_breaker.record_failure()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with graceful fallback"""
        with self.safe_operation("get") as client:
            if client:
                try:
                    result = client.get(key)
                    return result if result is not None else default
                except Exception:
                    pass
        return default
    
    def set(self, key: str, value: Any, ex: int = None) -> bool:
        """Set value with graceful fallback"""
        with self.safe_operation("set") as client:
            if client:
                try:
                    if ex:
                        return client.setex(key, ex, value)
                    else:
                        return client.set(key, value)
                except Exception:
                    pass
        return False
    
    def delete(self, *keys: str) -> int:
        """Delete keys with graceful fallback"""
        with self.safe_operation("delete") as client:
            if client:
                try:
                    return client.delete(*keys)
                except Exception:
                    pass
        return 0
    
    def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment with graceful fallback"""
        with self.safe_operation("incr") as client:
            if client:
                try:
                    return client.incr(key, amount)
                except Exception:
                    pass
        return None
    
    def expire(self, key: str, time: int) -> bool:
        """Set expiration with graceful fallback"""
        with self.safe_operation("expire") as client:
            if client:
                try:
                    return client.expire(key, time)
                except Exception:
                    pass
        return False
    
    def pipeline(self):
        """Get pipeline with graceful fallback"""
        if self._health_check() and self.circuit_breaker.can_execute():
            try:
                return self.client.pipeline()
            except Exception as e:
                logger.warning(f"Failed to create Redis pipeline: {e}")
                self.circuit_breaker.record_failure()
        return None
    
    def execute_pipeline(self, pipe, operation_name: str = "pipeline") -> Optional[list]:
        """Execute pipeline with error handling"""
        if not pipe:
            return None
        
        try:
            result = pipe.execute()
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            logger.warning(f"Redis pipeline execution failed in {operation_name}: {e}")
            self.circuit_breaker.record_failure()
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get Redis connection status"""
        return {
            "available": self.is_available,
            "circuit_state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count,
            "last_health_check": datetime.fromtimestamp(self.last_health_check).isoformat() if self.last_health_check else None,
            "redis_url_configured": bool(self.redis_url)
        }
    
    def force_reconnect(self):
        """Force reconnection attempt"""
        logger.info("Forcing Redis reconnection")
        self.circuit_breaker = RedisCircuitBreaker(self.circuit_breaker.config)
        self._initialize_connection()

# Global Redis manager instance
_redis_manager = None

def get_redis_manager() -> RedisManager:
    """Get global Redis manager instance"""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
    return _redis_manager

def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client with fallback handling"""
    manager = get_redis_manager()
    if manager.is_available and manager.circuit_breaker.can_execute():
        return manager.client
    return None

# Convenience functions for backward compatibility
def redis_get(key: str, default: Any = None) -> Any:
    """Get value from Redis with graceful fallback"""
    return get_redis_manager().get(key, default)

def redis_set(key: str, value: Any, ex: int = None) -> bool:
    """Set value in Redis with graceful fallback"""
    return get_redis_manager().set(key, value, ex)

def redis_delete(*keys: str) -> int:
    """Delete keys from Redis with graceful fallback"""
    return get_redis_manager().delete(*keys)

def redis_available() -> bool:
    """Check if Redis is available"""
    return get_redis_manager().is_available

def redis_status() -> Dict[str, Any]:
    """Get Redis status information"""
    return get_redis_manager().get_status()