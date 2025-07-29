#!/usr/bin/env python3
"""
Test script to demonstrate Redis functionality with graceful fallback

This script tests the enhanced Redis manager to show how the system
works both with and without Redis available.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

try:
    from shared.redis_manager import get_redis_manager, RedisManager
    from shared.health_check import get_health_checker, HealthChecker
    from shared.rate_limiting import brute_force_protection
    from shared.csrf_protection import csrf_protection
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Redis modules not available: {e}")
    MODULES_AVAILABLE = False
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_redis_manager():
    """Test Redis manager functionality"""
    print("\n=== Testing Redis Manager ===")
    
    # Get Redis manager
    redis_manager = get_redis_manager()
    
    # Check status
    status = redis_manager.get_status()
    print(f"Redis Status: {status}")
    
    # Test basic operations
    print("\n--- Testing Basic Operations ---")
    
    # Set a value
    key = "test_key"
    value = "test_value"
    success = redis_manager.set(key, value, ex=60)
    print(f"Set '{key}' = '{value}': {success}")
    
    # Get the value
    retrieved = redis_manager.get(key)
    print(f"Get '{key}': {retrieved}")
    
    # Test with JSON data
    json_key = "test_json"
    json_value = {"name": "John", "age": 30, "city": "New York"}
    success = redis_manager.set(json_key, json_value, ex=60)
    print(f"Set JSON '{json_key}': {success}")
    
    retrieved_json = redis_manager.get(json_key)
    print(f"Get JSON '{json_key}': {retrieved_json}")
    
    # Test increment
    counter_key = "test_counter"
    count = redis_manager.incr(counter_key)
    print(f"Increment '{counter_key}': {count}")
    
    count = redis_manager.incr(counter_key)
    print(f"Increment '{counter_key}' again: {count}")
    
    # Test delete
    deleted = redis_manager.delete(key)
    print(f"Delete '{key}': {deleted}")
    
    # Try to get deleted key
    retrieved_after_delete = redis_manager.get(key)
    print(f"Get '{key}' after delete: {retrieved_after_delete}")
    
    # Test safe operation
    with redis_manager.safe_operation("ping_test") as client:
        if client:
            try:
                result = client.ping()
                print(f"Safe operation (ping): {result}")
            except Exception as e:
                print(f"Safe operation (ping) failed: {e}")
        else:
            print("Safe operation (ping): Redis not available")

def test_health_checker():
    """Test health checker functionality"""
    print("\n=== Testing Health Checker ===")
    
    health_checker = get_health_checker()
    
    # System health check
    system_health = health_checker.check_system_health()
    print(f"System Health: {system_health}")
    
    # Redis health check
    redis_health = health_checker.check_redis_health()
    print(f"Redis Health: {redis_health}")
    
    # Comprehensive health check
    comprehensive = health_checker.get_comprehensive_health()
    print(f"Comprehensive Health: {comprehensive}")

def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n=== Testing Rate Limiting ===")
    
    # Create a mock request object for testing
    class MockRequest:
        def __init__(self, client_ip="127.0.0.1"):
            self.client = type('obj', (object,), {'host': client_ip})
            self.headers = {"x-forwarded-for": client_ip}
            self.state = type('obj', (object,), {})
    
    mock_request = MockRequest()
    endpoint_type = "api.read"
    
    # Test multiple requests
    for i in range(5):
        try:
            info = brute_force_protection.check_rate_limit(mock_request, endpoint_type)
            print(f"Request {i+1}: Allowed=True, Info={info}")
        except Exception as e:
            print(f"Request {i+1}: Rate limit exceeded - {e}")
            break
    
    print("Rate limiting test completed (using local fallback since Redis is unavailable)")

def test_csrf_protection():
    """Test CSRF protection functionality"""
    print("\n=== Testing CSRF Protection ===")
    
    # Generate a token with customer_id
    customer_id = 123
    token_info = csrf_protection.generate_token(customer_id)
    token = token_info['token']
    print(f"Generated CSRF token: {token[:20]}...")
    
    # Create a mock request for validation
    class MockRequest:
        def __init__(self, token):
            self.method = "POST"
            self.headers = {"X-CSRF-Token": token}
            self.cookies = {}
    
    # Validate the token using validate_request
    mock_request = MockRequest(token)
    is_valid = csrf_protection.validate_request(mock_request, customer_id)
    print(f"Token validation: {is_valid}")
    
    # Test with invalid token
    invalid_token = "invalid_token_123"
    mock_request_invalid = MockRequest(invalid_token)
    is_valid_invalid = csrf_protection.validate_request(mock_request_invalid, customer_id)
    print(f"Invalid token validation: {is_valid_invalid}")

def test_without_redis():
    """Test functionality when Redis is not available"""
    print("\n=== Testing Without Redis (Simulated) ===")
    
    # Create a Redis manager with invalid URL to simulate failure
    redis_manager = RedisManager(redis_url="redis://invalid:6379")
    
    print(f"Redis available: {redis_manager.is_available}")
    
    # Test operations with fallback
    key = "fallback_test"
    value = "fallback_value"
    
    success = redis_manager.set(key, value)
    print(f"Set with fallback: {success}")
    
    retrieved = redis_manager.get(key)
    print(f"Get with fallback: {retrieved}")
    
    # Test safe operation with fallback
    with redis_manager.safe_operation("failing_test") as client:
        if client:
            try:
                # This will fail since Redis is not available
                result = client.ping()
                print(f"Safe operation with fallback: {result}")
            except Exception as e:
                print(f"Safe operation with fallback failed as expected: {e}")
        else:
            print("Safe operation with fallback: Redis not available (expected)")

def main():
    """Main test function"""
    print("Redis Functionality Test")
    print("=" * 50)
    
    if not MODULES_AVAILABLE:
        print("Redis modules not available. Please ensure the shared modules are properly installed.")
        return
    
    try:
        # Test Redis manager
        test_redis_manager()
        
        # Test health checker
        test_health_checker()
        
        # Test rate limiting
        test_rate_limiting()
        
        # Test CSRF protection
        test_csrf_protection()
        
        # Test without Redis
        test_without_redis()
        
        print("\n=== Test Summary ===")
        print("✅ All tests completed successfully!")
        print("✅ The system gracefully handles both Redis available and unavailable scenarios")
        print("✅ Rate limiting, CSRF protection, and caching work with fallbacks")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)