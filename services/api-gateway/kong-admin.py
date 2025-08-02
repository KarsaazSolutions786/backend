#!/usr/bin/env python3
"""
Kong API Gateway Management Script
Provides easy management of Kong services, routes, and plugins
"""

import requests
import json
import argparse
import time
import sys
from typing import Dict, List, Optional

class KongAdmin:
    def __init__(self, kong_admin_url: str = "http://localhost:8101"):
        self.kong_admin_url = kong_admin_url.rstrip('/')
        
    def wait_for_kong(self, timeout: int = 60) -> bool:
        """Wait for Kong to be ready"""
        print("⏳ Waiting for Kong to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.kong_admin_url}/status", timeout=5)
                if response.status_code == 200:
                    print("✅ Kong is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print("Kong not ready yet, waiting...")
            time.sleep(5)
        
        print("❌ Kong failed to start within timeout")
        return False
    
    def create_service(self, name: str, url: str, **kwargs) -> Dict:
        """Create a Kong service"""
        data = {
            "name": name,
            "url": url,
            "retries": kwargs.get("retries", 3),
            "connect_timeout": kwargs.get("connect_timeout", 60000),
            "write_timeout": kwargs.get("write_timeout", 60000),
            "read_timeout": kwargs.get("read_timeout", 60000)
        }
        
        response = requests.post(f"{self.kong_admin_url}/services/", data=data)
        if response.status_code == 201:
            print(f"✅ Service '{name}' created successfully")
            return response.json()
        elif response.status_code == 409:
            print(f"⚠️  Service '{name}' already exists")
            return self.get_service(name)
        else:
            print(f"❌ Failed to create service '{name}': {response.text}")
            return {}
    
    def create_route(self, service_name: str, paths: List[str], **kwargs) -> Dict:
        """Create a route for a service"""
        data = {
            "paths": paths,
            "strip_path": kwargs.get("strip_path", False),
            "preserve_host": kwargs.get("preserve_host", False)
        }
        
        response = requests.post(f"{self.kong_admin_url}/services/{service_name}/routes", data=data)
        if response.status_code == 201:
            print(f"✅ Route for service '{service_name}' created successfully")
            return response.json()
        elif response.status_code == 409:
            print(f"⚠️  Route for service '{service_name}' already exists")
            return {}
        else:
            print(f"❌ Failed to create route for service '{service_name}': {response.text}")
            return {}
    
    def add_plugin(self, service_name: str, plugin_name: str, config: Dict = None) -> Dict:
        """Add a plugin to a service"""
        data = {"name": plugin_name}
        if config:
            for key, value in config.items():
                if isinstance(value, (list, dict)):
                    data[f"config.{key}"] = json.dumps(value) if isinstance(value, dict) else value
                else:
                    data[f"config.{key}"] = value
        
        response = requests.post(f"{self.kong_admin_url}/services/{service_name}/plugins", data=data)
        if response.status_code == 201:
            print(f"✅ Plugin '{plugin_name}' added to service '{service_name}'")
            return response.json()
        elif response.status_code == 409:
            print(f"⚠️  Plugin '{plugin_name}' already exists on service '{service_name}'")
            return {}
        else:
            print(f"❌ Failed to add plugin '{plugin_name}' to service '{service_name}': {response.text}")
            return {}
    
    def add_global_plugin(self, plugin_name: str, config: Dict = None) -> Dict:
        """Add a global plugin"""
        data = {"name": plugin_name}
        if config:
            for key, value in config.items():
                data[f"config.{key}"] = value
        
        response = requests.post(f"{self.kong_admin_url}/plugins/", data=data)
        if response.status_code == 201:
            print(f"✅ Global plugin '{plugin_name}' added")
            return response.json()
        elif response.status_code == 409:
            print(f"⚠️  Global plugin '{plugin_name}' already exists")
            return {}
        else:
            print(f"❌ Failed to add global plugin '{plugin_name}': {response.text}")
            return {}
    
    def create_consumer(self, username: str) -> Dict:
        """Create a consumer"""
        data = {"username": username}
        
        response = requests.post(f"{self.kong_admin_url}/consumers/", data=data)
        if response.status_code == 201:
            print(f"✅ Consumer '{username}' created")
            return response.json()
        elif response.status_code == 409:
            print(f"⚠️  Consumer '{username}' already exists")
            return self.get_consumer(username)
        else:
            print(f"❌ Failed to create consumer '{username}': {response.text}")
            return {}
    
    def add_jwt_to_consumer(self, username: str, key: str, secret: str) -> Dict:
        """Add JWT credentials to a consumer"""
        data = {"key": key, "secret": secret}
        
        response = requests.post(f"{self.kong_admin_url}/consumers/{username}/jwt", data=data)
        if response.status_code == 201:
            print(f"✅ JWT credentials added to consumer '{username}'")
            return response.json()
        elif response.status_code == 409:
            print(f"⚠️  JWT credentials already exist for consumer '{username}'")
            return {}
        else:
            print(f"❌ Failed to add JWT credentials to consumer '{username}': {response.text}")
            return {}
    
    def get_service(self, name: str) -> Dict:
        """Get service by name"""
        response = requests.get(f"{self.kong_admin_url}/services/{name}")
        return response.json() if response.status_code == 200 else {}
    
    def get_consumer(self, username: str) -> Dict:
        """Get consumer by username"""
        response = requests.get(f"{self.kong_admin_url}/consumers/{username}")
        return response.json() if response.status_code == 200 else {}
    
    def list_services(self) -> List[Dict]:
        """List all services"""
        response = requests.get(f"{self.kong_admin_url}/services/")
        return response.json().get("data", []) if response.status_code == 200 else []
    
    def list_routes(self) -> List[Dict]:
        """List all routes"""
        response = requests.get(f"{self.kong_admin_url}/routes/")
        return response.json().get("data", []) if response.status_code == 200 else []
    
    def list_plugins(self) -> List[Dict]:
        """List all plugins"""
        response = requests.get(f"{self.kong_admin_url}/plugins/")
        return response.json().get("data", []) if response.status_code == 200 else []

def setup_microservices_gateway():
    """Setup complete microservices gateway configuration"""
    kong = KongAdmin()
    
    if not kong.wait_for_kong():
        sys.exit(1)
    
    print("🚀 Setting up microservices gateway...")
    
    # Service configurations
    services = [
        {
            "name": "auth-service",
            "url": "http://auth-service:8000",
            "paths": ["/auth"],
            "jwt_required": False,  # Auth service doesn't require JWT
            "rate_limits": {"minute": 60, "hour": 600}
        },
        {
            "name": "user-service",
            "url": "http://user-service:8000",
            "paths": ["/users"],
            "jwt_required": True,
            "rate_limits": {"minute": 100, "hour": 1000}
        },
        {
            "name": "reminder-service",
            "url": "http://reminder-service:8000",
            "paths": ["/reminders"],
            "jwt_required": True,
            "rate_limits": {"minute": 150, "hour": 1500}
        },
        {
            "name": "note-service",
            "url": "http://note-service:8000",
            "paths": ["/notes"],
            "jwt_required": True,
            "rate_limits": {"minute": 200, "hour": 2000}
        },
        {
            "name": "ledger-service",
            "url": "http://ledger-service:8000",
            "paths": ["/expenses"],
            "jwt_required": True,
            "rate_limits": {"minute": 100, "hour": 1000}
        },
        {
            "name": "friend-service",
            "url": "http://friend-service:8000",
            "paths": ["/friends"],
            "jwt_required": True,
            "rate_limits": {"minute": 50, "hour": 500}
        },
        {
            "name": "history-service",
            "url": "http://history-service:8000",
            "paths": ["/logs"],
            "jwt_required": True,
            "rate_limits": {"minute": 50, "hour": 500}
        },
        {
            "name": "stt-service",
            "url": "http://stt-service:8000",
            "paths": ["/stt"],
            "jwt_required": True,
            "rate_limits": {"minute": 30, "hour": 300}
        },
        {
            "name": "tts-service",
            "url": "http://tts-service:8000",
            "paths": ["/tts"],
            "jwt_required": True,
            "rate_limits": {"minute": 30, "hour": 300}
        },
        {
            "name": "intent-service",
            "url": "http://intent-service:8000",
            "paths": ["/intent"],
            "jwt_required": True,
            "rate_limits": {"minute": 100, "hour": 1000}
        },
        {
            "name": "chat-service",
            "url": "http://chat-service:8000",
            "paths": ["/conversations"],
            "jwt_required": True,
            "rate_limits": {"minute": 200, "hour": 2000}
        },
        {
            "name": "scheduler-service",
            "url": "http://scheduler-service:8000",
            "paths": ["/jobs"],
            "jwt_required": True,
            "rate_limits": {"minute": 50, "hour": 500}
        },
        {
            "name": "health-service",
            "url": "http://auth-service:8000",
            "paths": ["/health"],
            "jwt_required": False,
            "rate_limits": {"minute": 100, "hour": 1000}
        }
    ]
    
    # Create services and routes
    for service_config in services:
        # Create service
        kong.create_service(
            name=service_config["name"],
            url=service_config["url"]
        )
        
        # Create route
        kong.create_route(
            service_name=service_config["name"],
            paths=service_config["paths"]
        )
        
        # Add CORS plugin
        kong.add_plugin(
            service_name=service_config["name"],
            plugin_name="cors",
            config={
                "origins": ["*"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
                "headers": ["Accept", "Accept-Version", "Content-Length", "Content-MD5", 
                           "Content-Type", "Date", "Authorization"],
                "exposed_headers": ["X-Auth-Token"],
                "credentials": True,
                "max_age": 3600
            }
        )
        
        # Add JWT plugin if required
        if service_config["jwt_required"]:
            kong.add_plugin(
                service_name=service_config["name"],
                plugin_name="jwt",
                config={
                    "secret_is_base64": False,
                    "key_claim_name": "iss",
                    "claims_to_verify": ["exp"]
                }
            )
        
        # Add rate limiting
        kong.add_plugin(
            service_name=service_config["name"],
            plugin_name="rate-limiting",
            config={
                "minute": service_config["rate_limits"]["minute"],
                "hour": service_config["rate_limits"]["hour"],
                "policy": "local"
            }
        )
    
    # Create JWT consumer for authentication
    kong.create_consumer("eindr-auth-service")
    kong.add_jwt_to_consumer(
        username="eindr-auth-service",
        key="eindr-issuer",  # Must match the 'iss' claim in JWT tokens
        secret="eindr-super-secure-jwt-secret-key-for-production-2024-v1"
    )
    
    # Add global plugins
    kong.add_global_plugin(
        plugin_name="correlation-id",
        config={
            "header_name": "X-Request-ID",
            "generator": "uuid"
        }
    )
    
    kong.add_global_plugin(
        plugin_name="file-log",
        config={
            "path": "/tmp/access.log"
        }
    )
    
    print("🎉 Kong setup completed!")
    print()
    print("📋 Available endpoints:")
    print("🌐 API Gateway: http://localhost:8080")
    print("⚙️  Kong Admin API: http://localhost:8101")
    print("📊 Kong Admin GUI: http://localhost:8102")
    print("🖥️  Konga UI: http://localhost:8103")

def main():
    parser = argparse.ArgumentParser(description="Kong API Gateway Management")
    parser.add_argument("command", choices=["setup", "status", "list"], help="Command to execute")
    parser.add_argument("--kong-url", default="http://localhost:8101", help="Kong Admin API URL")
    
    args = parser.parse_args()
    
    kong = KongAdmin(args.kong_url)
    
    if args.command == "setup":
        setup_microservices_gateway()
    elif args.command == "status":
        if kong.wait_for_kong(timeout=10):
            print("✅ Kong is running")
        else:
            print("❌ Kong is not accessible")
    elif args.command == "list":
        services = kong.list_services()
        routes = kong.list_routes()
        plugins = kong.list_plugins()
        
        print(f"📊 Services: {len(services)}")
        for service in services:
            print(f"  - {service['name']}: {service['host']}")
        
        print(f"🛣️  Routes: {len(routes)}")
        for route in routes:
            print(f"  - {route.get('paths', [])}")
        
        print(f"🔌 Plugins: {len(plugins)}")
        for plugin in plugins:
            print(f"  - {plugin['name']}")

if __name__ == "__main__":
    main()