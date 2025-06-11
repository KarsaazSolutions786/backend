#!/usr/bin/env python3
"""
Simple script to generate a valid JWT token for testing the API in development mode.
"""

import jwt
import json
from datetime import datetime, timedelta

def generate_test_token():
    """Generate a simple JWT token for testing."""
    payload = {
        "sub": "test_user_123",  # Firebase UID
        "email": "test@example.com",
        "name": "Test User",
        "iat": int(datetime.utcnow().timestamp()),
        "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp())
    }
    
    # Create token without signature (development mode doesn't verify it)
    token = jwt.encode(payload, "test-secret", algorithm="HS256")
    return token

if __name__ == "__main__":
    token = generate_test_token()
    print("Generated test token:")
    print(token)
    
    # Verify the token can be decoded
    decoded = jwt.decode(token, options={"verify_signature": False})
    print("\nDecoded payload:")
    print(json.dumps(decoded, indent=2)) 