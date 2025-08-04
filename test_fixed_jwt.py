#!/usr/bin/env python3
import jwt
import requests
from datetime import datetime, timedelta

# Configuration
SECRET_KEY = "eindr-super-secure-jwt-secret-key-for-production-2024-v1"
ALGORITHM = "HS256"
CUSTOMER_SERVICE_URL = "http://localhost:8002"

def create_jwt_token_with_audience():
    """Create a JWT token with proper audience and issuer claims matching auth service format"""
    payload = {
        "sub": "1",  # Using customer ID 1 (ashhad@example.com)
        "email": "ashhad@example.com",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=1),
        "type": "access",  # Required by auth service
        "aud": "eindr-api",  # Audience claim
        "iss": "eindr-issuer",  # Issuer claim
        "permissions": ["user"]  # Basic permissions
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    print(f"Generated token: {token[:50]}...")
    return token

def test_customer_endpoint():
    """Test the customer service /customers/me endpoint"""
    token = create_jwt_token_with_audience()
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(f"{CUSTOMER_SERVICE_URL}/customers/me", headers=headers)
        print(f"Response Status: {response.status_code}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            print("✅ JWT validation successful!")
        else:
            print("❌ JWT validation failed")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    print("Testing JWT validation with audience and issuer claims...")
    test_customer_endpoint()