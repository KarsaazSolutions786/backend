#!/usr/bin/env python3
"""
Model Verification Script for Eindr Backend
Verifies that AI models are properly distributed across services
"""

import os
import sys
from pathlib import Path

def format_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def check_git_lfs_pointer(file_path):
    """Check if file is a Git LFS pointer"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
                return True
    except:
        pass
    return False

def verify_models():
    """Verify model files in both global and service-specific locations"""
    
    # Define expected models and their locations
    models_config = {
        "whisper-tiny.bin": {
            "description": "OpenAI Whisper STT Model",
            "service": "stt-service",
            "expected_size_mb": 151
        },
        "coqui.tflite": {
            "description": "Coqui TTS Model",
            "service": "tts-service", 
            "expected_size_mb": 45
        },
        "Mini_LM.bin": {
            "description": "MiniLM Intent Classification Model",
            "service": "intent-service",
            "expected_size_mb": 91
        }
    }
    
    print("🤖 Eindr Backend - AI Models Verification")
    print("=" * 50)
    
    # Check global models directory
    print("\n📁 Global Models Directory (./models/)")
    global_models_dir = Path("models")
    
    if not global_models_dir.exists():
        print("❌ Global models directory not found!")
        global_models_exist = False
    else:
        global_models_exist = True
        for model_file in global_models_dir.iterdir():
            if model_file.is_file():
                size = model_file.stat().st_size
                is_lfs = check_git_lfs_pointer(model_file)
                status = "🔗 Git LFS Pointer" if is_lfs else "✅ Actual File"
                print(f"  {status}: {model_file.name} ({format_size(size)})")
    
    # Check service-specific models
    print("\n🏢 Service-Specific Models")
    print("-" * 30)
    
    all_services_ok = True
    
    for model_name, config in models_config.items():
        service_name = config["service"]
        description = config["description"]
        expected_size_mb = config["expected_size_mb"]
        
        print(f"\n📦 {service_name.upper()}")
        print(f"   Model: {description}")
        
        # Check service model directory
        service_model_path = Path(f"services/{service_name}/models/{model_name}")
        
        if service_model_path.exists():
            size = service_model_path.stat().st_size
            size_mb = size / (1024 * 1024)
            is_lfs = check_git_lfs_pointer(service_model_path)
            
            if is_lfs:
                print(f"   ⚠️  Found Git LFS pointer ({format_size(size)})")
                print(f"      This is expected if using Git LFS")
            elif size_mb >= expected_size_mb * 0.8:  # Allow 20% variance
                print(f"   ✅ Model file present ({format_size(size)})")
            else:
                print(f"   ⚠️  Model file small ({format_size(size)}) - expected ~{expected_size_mb}MB")
                
        else:
            print(f"   ❌ Model file missing: {service_model_path}")
            all_services_ok = False
            
        # Check if service directory exists
        service_dir = Path(f"services/{service_name}")
        if not service_dir.exists():
            print(f"   ❌ Service directory missing: {service_dir}")
            all_services_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    if global_models_exist:
        print("✅ Global models directory exists")
    else:
        print("❌ Global models directory missing")
    
    if all_services_ok:
        print("✅ All service model files detected")
    else:
        print("⚠️  Some service model files may be missing or are Git LFS pointers")
    
    # Docker setup check
    print("\n🐳 Docker Configuration")
    compose_file = Path("docker-compose.microservices.yml")
    if compose_file.exists():
        print("✅ Docker compose file found")
        
        # Quick check for model volumes
        with open(compose_file, 'r') as f:
            content = f.read()
            if "./models:/app/models:ro" in content:
                print("✅ Global model volumes configured")
            else:
                print("⚠️  Global model volumes may not be configured")
                
            ai_services = ["stt-service", "tts-service", "intent-service"]
            configured_services = 0
            for service in ai_services:
                if f"{service}:" in content:
                    configured_services += 1
            
            print(f"✅ {configured_services}/{len(ai_services)} AI services configured in Docker")
    else:
        print("❌ Docker compose file not found")
    
    # Usage instructions
    print("\n📚 USAGE INSTRUCTIONS")
    print("-" * 25)
    print("To use the AI models:")
    print("1. Ensure models are available (either actual files or Git LFS)")
    print("2. Run: make up  # Start all services with Docker")
    print("3. Check health:")
    print("   curl http://localhost:8008/stt/health")
    print("   curl http://localhost:8009/tts/health") 
    print("   curl http://localhost:8010/intent/health")
    
    if not all_services_ok or not global_models_exist:
        print("\n⚠️  If models appear as Git LFS pointers:")
        print("   git lfs pull  # Download actual model files")
        
    print("\n🎯 Model Loading Priority:")
    print("   1. Service local models (./services/{service}/models/)")
    print("   2. Environment variable ($MODEL_PATH)")
    print("   3. Global models (./models/ via Docker)")
    print("   4. Standard model downloads (fallback)")

if __name__ == "__main__":
    try:
        verify_models()
    except KeyboardInterrupt:
        print("\n\n⏹️  Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        sys.exit(1) 
"""
Model Verification Script for Eindr Backend
Verifies that AI models are properly distributed across services
"""

import os
import sys
from pathlib import Path

def format_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def check_git_lfs_pointer(file_path):
    """Check if file is a Git LFS pointer"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
                return True
    except:
        pass
    return False

def verify_models():
    """Verify model files in both global and service-specific locations"""
    
    # Define expected models and their locations
    models_config = {
        "whisper-tiny.bin": {
            "description": "OpenAI Whisper STT Model",
            "service": "stt-service",
            "expected_size_mb": 151
        },
        "coqui.tflite": {
            "description": "Coqui TTS Model",
            "service": "tts-service", 
            "expected_size_mb": 45
        },
        "Mini_LM.bin": {
            "description": "MiniLM Intent Classification Model",
            "service": "intent-service",
            "expected_size_mb": 91
        }
    }
    
    print("🤖 Eindr Backend - AI Models Verification")
    print("=" * 50)
    
    # Check global models directory
    print("\n📁 Global Models Directory (./models/)")
    global_models_dir = Path("models")
    
    if not global_models_dir.exists():
        print("❌ Global models directory not found!")
        global_models_exist = False
    else:
        global_models_exist = True
        for model_file in global_models_dir.iterdir():
            if model_file.is_file():
                size = model_file.stat().st_size
                is_lfs = check_git_lfs_pointer(model_file)
                status = "🔗 Git LFS Pointer" if is_lfs else "✅ Actual File"
                print(f"  {status}: {model_file.name} ({format_size(size)})")
    
    # Check service-specific models
    print("\n🏢 Service-Specific Models")
    print("-" * 30)
    
    all_services_ok = True
    
    for model_name, config in models_config.items():
        service_name = config["service"]
        description = config["description"]
        expected_size_mb = config["expected_size_mb"]
        
        print(f"\n📦 {service_name.upper()}")
        print(f"   Model: {description}")
        
        # Check service model directory
        service_model_path = Path(f"services/{service_name}/models/{model_name}")
        
        if service_model_path.exists():
            size = service_model_path.stat().st_size
            size_mb = size / (1024 * 1024)
            is_lfs = check_git_lfs_pointer(service_model_path)
            
            if is_lfs:
                print(f"   ⚠️  Found Git LFS pointer ({format_size(size)})")
                print(f"      This is expected if using Git LFS")
            elif size_mb >= expected_size_mb * 0.8:  # Allow 20% variance
                print(f"   ✅ Model file present ({format_size(size)})")
            else:
                print(f"   ⚠️  Model file small ({format_size(size)}) - expected ~{expected_size_mb}MB")
                
        else:
            print(f"   ❌ Model file missing: {service_model_path}")
            all_services_ok = False
            
        # Check if service directory exists
        service_dir = Path(f"services/{service_name}")
        if not service_dir.exists():
            print(f"   ❌ Service directory missing: {service_dir}")
            all_services_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    if global_models_exist:
        print("✅ Global models directory exists")
    else:
        print("❌ Global models directory missing")
    
    if all_services_ok:
        print("✅ All service model files detected")
    else:
        print("⚠️  Some service model files may be missing or are Git LFS pointers")
    
    # Docker setup check
    print("\n🐳 Docker Configuration")
    compose_file = Path("docker-compose.microservices.yml")
    if compose_file.exists():
        print("✅ Docker compose file found")
        
        # Quick check for model volumes
        with open(compose_file, 'r') as f:
            content = f.read()
            if "./models:/app/models:ro" in content:
                print("✅ Global model volumes configured")
            else:
                print("⚠️  Global model volumes may not be configured")
                
            ai_services = ["stt-service", "tts-service", "intent-service"]
            configured_services = 0
            for service in ai_services:
                if f"{service}:" in content:
                    configured_services += 1
            
            print(f"✅ {configured_services}/{len(ai_services)} AI services configured in Docker")
    else:
        print("❌ Docker compose file not found")
    
    # Usage instructions
    print("\n📚 USAGE INSTRUCTIONS")
    print("-" * 25)
    print("To use the AI models:")
    print("1. Ensure models are available (either actual files or Git LFS)")
    print("2. Run: make up  # Start all services with Docker")
    print("3. Check health:")
    print("   curl http://localhost:8008/stt/health")
    print("   curl http://localhost:8009/tts/health") 
    print("   curl http://localhost:8010/intent/health")
    
    if not all_services_ok or not global_models_exist:
        print("\n⚠️  If models appear as Git LFS pointers:")
        print("   git lfs pull  # Download actual model files")
        
    print("\n🎯 Model Loading Priority:")
    print("   1. Service local models (./services/{service}/models/)")
    print("   2. Environment variable ($MODEL_PATH)")
    print("   3. Global models (./models/ via Docker)")
    print("   4. Standard model downloads (fallback)")

if __name__ == "__main__":
    try:
        verify_models()
    except KeyboardInterrupt:
        print("\n\n⏹️  Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        sys.exit(1) 
 
 