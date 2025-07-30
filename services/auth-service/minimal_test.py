#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Simple test models
class LoginRequest(BaseModel):
    email: str
    password: str
    remember_me: bool = False

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    message: str

# Create minimal FastAPI app
app = FastAPI(title="Auth Service Test")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "Auth service test is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/auth/login")
async def login(login_data: LoginRequest):
    """Test login endpoint"""
    try:
        # Simple validation
        if not login_data.email or not login_data.password:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        # For testing, accept any email/password combination
        if "@" not in login_data.email:
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Return success response
        return LoginResponse(
            access_token="test_token_12345",
            message="Login successful (test mode)"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login temporarily unavailable. Please try again later.")

if __name__ == "__main__":
    print("Starting minimal auth service test on port 8003...")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")