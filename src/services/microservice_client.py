"""
Microservice Client for communicating with actual microservices
"""

import httpx
import os
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class MicroserviceClient:
    """Client for communicating with microservices"""
    
    def __init__(self):
        # Get service URLs from environment variables
        self.auth_service_url = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
        self.customer_service_url = os.getenv("CUSTOMER_SERVICE_URL", "http://localhost:8002")
        self.reminder_service_url = os.getenv("REMINDER_SERVICE_URL", "http://localhost:8003")
        self.note_service_url = os.getenv("NOTE_SERVICE_URL", "http://localhost:8004")
        self.ledger_service_url = os.getenv("LEDGER_SERVICE_URL", "http://localhost:8005")
        self.friend_service_url = os.getenv("FRIEND_SERVICE_URL", "http://localhost:8006")
        self.chat_service_url = os.getenv("CHAT_SERVICE_URL", "http://localhost:8011")
        
        # Timeout for requests
        self.timeout = 30.0
        
    async def _make_request(
        self, 
        method: str, 
        service_url: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to microservice"""
        try:
            url = f"{service_url}{endpoint}"
            
            # Add authorization header if token provided
            if token and headers is None:
                headers = {"Authorization": f"Bearer {token}"}
            elif token:
                headers["Authorization"] = f"Bearer {token}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data, headers=headers)
                elif method.upper() == "PUT":
                    response = await client.put(url, json=data, headers=headers)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
        except httpx.TimeoutException:
            logger.error(f"Timeout connecting to {service_url}")
            raise HTTPException(status_code=503, detail="Service timeout")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from {service_url}: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail="Service error")
        except Exception as e:
            logger.error(f"Error connecting to {service_url}: {str(e)}")
            raise HTTPException(status_code=503, detail="Service unavailable")

    # Auth Service Methods
    async def register_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register user via auth service"""
        return await self._make_request("POST", self.auth_service_url, "/register", data=user_data)
    
    async def login_user(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Login user via auth service"""
        return await self._make_request("POST", self.auth_service_url, "/login", data=credentials)
    
    async def get_user_info(self, token: str) -> Dict[str, Any]:
        """Get user info via auth service"""
        return await self._make_request("GET", self.auth_service_url, "/me", token=token)

    # Customer Service Methods
    async def get_customers(self, token: str) -> Dict[str, Any]:
        """Get customers via customer service"""
        return await self._make_request("GET", self.customer_service_url, "/customers", token=token)
    
    async def get_customer(self, customer_id: int, token: str) -> Dict[str, Any]:
        """Get customer by ID via customer service"""
        return await self._make_request("GET", self.customer_service_url, f"/customers/{customer_id}", token=token)
    
    async def create_customer(self, customer_data: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Create customer via customer service"""
        return await self._make_request("POST", self.customer_service_url, "/customers", data=customer_data, token=token)

    # Reminder Service Methods
    async def get_reminders(self, token: str) -> Dict[str, Any]:
        """Get reminders via reminder service"""
        return await self._make_request("GET", self.reminder_service_url, "/reminders", token=token)
    
    async def create_reminder(self, reminder_data: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Create reminder via reminder service"""
        return await self._make_request("POST", self.reminder_service_url, "/reminders", data=reminder_data, token=token)

    # Note Service Methods
    async def get_notes(self, token: str) -> Dict[str, Any]:
        """Get notes via note service"""
        return await self._make_request("GET", self.note_service_url, "/notes", token=token)
    
    async def create_note(self, note_data: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Create note via note service"""
        return await self._make_request("POST", self.note_service_url, "/notes", data=note_data, token=token)

    # Ledger Service Methods
    async def get_ledger_entries(self, token: str) -> Dict[str, Any]:
        """Get ledger entries via ledger service"""
        return await self._make_request("GET", self.ledger_service_url, "/ledger", token=token)
    
    async def create_ledger_entry(self, entry_data: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Create ledger entry via ledger service"""
        return await self._make_request("POST", self.ledger_service_url, "/ledger", data=entry_data, token=token)

    # Friend Service Methods
    async def get_friends(self, token: str) -> Dict[str, Any]:
        """Get friends via friend service"""
        return await self._make_request("GET", self.friend_service_url, "/friends", token=token)
    
    async def add_friend(self, friend_data: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Add friend via friend service"""
        return await self._make_request("POST", self.friend_service_url, "/friends", data=friend_data, token=token)

    # Chat Service Methods
    async def chat(self, message_data: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Chat via chat service"""
        return await self._make_request("POST", self.chat_service_url, "/chat", data=message_data, token=token)

# Global client instance
microservice_client = MicroserviceClient() 