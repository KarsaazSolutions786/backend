"""Health Check Module for Eindr Backend Services

Provides comprehensive health checks for Redis, database, and other critical services.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .redis_manager import get_redis_manager

logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checker for backend services"""
    
    def __init__(self):
        self.redis_manager = get_redis_manager()
        self.start_time = time.time()
    
    def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connection and performance"""
        redis_status = self.redis_manager.get_status()
        
        # Additional performance test
        test_key = "health_check_test"
        test_value = str(time.time())
        
        performance_test = {
            "write_test": False,
            "read_test": False,
            "delete_test": False,
            "response_time_ms": None
        }
        
        if redis_status["available"]:
            start_time = time.time()
            
            # Test write
            if self.redis_manager.set(test_key, test_value, ex=60):
                performance_test["write_test"] = True
                
                # Test read
                retrieved_value = self.redis_manager.get(test_key)
                if retrieved_value == test_value:
                    performance_test["read_test"] = True
                    
                    # Test delete
                    if self.redis_manager.delete(test_key) > 0:
                        performance_test["delete_test"] = True
            
            performance_test["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return {
            **redis_status,
            "performance_test": performance_test,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        uptime_seconds = time.time() - self.start_time
        
        return {
            "status": "healthy",
            "uptime_seconds": round(uptime_seconds, 2),
            "uptime_human": self._format_uptime(uptime_seconds),
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",  # You can make this dynamic
            "environment": "development"  # You can read from env vars
        }
    
    def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status for all services"""
        redis_health = self.check_redis_health()
        system_health = self.check_system_health()
        
        # Determine overall status
        overall_status = "healthy"
        if not redis_health["available"]:
            overall_status = "degraded"  # Still functional with fallbacks
        
        issues = []
        if not redis_health["available"]:
            issues.append("Redis connection unavailable - using local fallbacks")
        
        if redis_health["circuit_state"] == "open":
            issues.append("Redis circuit breaker is open")
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "redis": redis_health,
                "system": system_health
            },
            "issues": issues,
            "summary": {
                "redis_available": redis_health["available"],
                "redis_circuit_state": redis_health["circuit_state"],
                "uptime_seconds": system_health["uptime_seconds"]
            }
        }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def force_redis_reconnect(self) -> Dict[str, Any]:
        """Force Redis reconnection and return status"""
        logger.info("Forcing Redis reconnection via health checker")
        self.redis_manager.force_reconnect()
        return self.check_redis_health()

# Global health checker instance
_health_checker = None

def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

# Convenience functions
def check_health() -> Dict[str, Any]:
    """Quick health check"""
    return get_health_checker().get_comprehensive_health()

def check_redis() -> Dict[str, Any]:
    """Quick Redis health check"""
    return get_health_checker().check_redis_health()

def force_redis_reconnect() -> Dict[str, Any]:
    """Force Redis reconnection"""
    return get_health_checker().force_redis_reconnect()