#!/usr/bin/env python3
import requests
import json
import sys

def test_auth_and_notes():
    """Test authentication and notes API"""
    
    # API endpoints
    auth_url = "https://auth-production-37f1.up.railway.app/auth/login"
    notes_url = "https://note-service-production.up.railway.app/notes"
    
    print("🔐 Testing authentication...")
    auth_data = {
        "email": "syedashhad17@gmail.com",
        "password": "Ashhad123",
        "remember_me": False
    }
    
    try:
        # Test login
        print(f"POST {auth_url}")
        auth_response = requests.post(auth_url, json=auth_data, timeout=30)
        print(f"Auth Status Code: {auth_response.status_code}")
        
        if auth_response.status_code == 200:
            auth_result = auth_response.json()
            print("✅ Authentication successful!")
            
            # Extract token info
            access_token = auth_result.get('access_token')
            if access_token:
                print(f"Access token preview: {access_token[:50]}...")
                
                # Decode token payload for debugging (without verification)
                import jwt
                try:
                    payload = jwt.decode(access_token, options={"verify_signature": False})
                    print(f"Token payload: {json.dumps(payload, indent=2)}")
                except Exception as e:
                    print(f"Could not decode token: {e}")
            
            # Test notes API
            if access_token:
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                
                print("\n📝 Testing notes API...")
                print(f"GET {notes_url}")
                notes_response = requests.get(notes_url, headers=headers, timeout=30)
                print(f"Notes API Status Code: {notes_response.status_code}")
                
                if notes_response.status_code == 200:
                    notes_result = notes_response.json()
                    print("✅ Notes API successful!")
                    print(f"Notes response: {json.dumps(notes_result, indent=2)}")
                    
                    # Test creating a note
                    print("\n📝 Testing note creation...")
                    note_data = {
                        "title": "Test Note",
                        "description": "This is a test note created via API",
                        "content_type": "text",
                        "is_favorite": False,
                        "is_pinned": False
                    }
                    
                    create_response = requests.post(notes_url + "/", json=note_data, headers=headers, timeout=30)
                    print(f"Create Note Status Code: {create_response.status_code}")
                    
                    if create_response.status_code == 201:
                        created_note = create_response.json()
                        print("✅ Note creation successful!")
                        print(f"Created note: {json.dumps(created_note, indent=2)}")
                    else:
                        print(f"❌ Note creation failed: {create_response.text}")
                        
                else:
                    print(f"❌ Notes API Error: {notes_response.text}")
                    
            else:
                print("❌ No access token received")
                
        else:
            print(f"❌ Authentication failed: {auth_response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_auth_and_notes()