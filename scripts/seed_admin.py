#!/usr/bin/env python3
"""
Admin Panel Seed Script
=======================

Initialize the admin panel with the first super admin user.
This script creates the initial super admin account from environment variables.

Usage:
    python scripts/seed_admin.py

Environment Variables:
    FIRST_SUPERADMIN_EMAIL - Email for the first super admin
    FIRST_SUPERADMIN_PASSWORD - Password for the first super admin
    FIRST_SUPERADMIN_NAME - Name for the first super admin (optional)
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from sqlalchemy import select
from connect_db import SessionLocal
from models.admin_models import AdminUser, AdminRole
from core.admin_security import validate_password_strength
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_first_super_admin():
    """Create the first super admin user from environment variables."""
    
    # Get configuration from environment
    email = os.getenv("FIRST_SUPERADMIN_EMAIL")
    password = os.getenv("FIRST_SUPERADMIN_PASSWORD") 
    name = os.getenv("FIRST_SUPERADMIN_NAME", "Super Administrator")
    
    if not email or not password:
        print("❌ Error: FIRST_SUPERADMIN_EMAIL and FIRST_SUPERADMIN_PASSWORD environment variables are required")
        print("\nUsage:")
        print("export FIRST_SUPERADMIN_EMAIL='admin@yourcompany.com'")
        print("export FIRST_SUPERADMIN_PASSWORD='your_secure_password'")
        print("export FIRST_SUPERADMIN_NAME='Your Name'  # Optional")
        print("python scripts/seed_admin.py")
        return False
    
    # Validate email format
    if "@" not in email or "." not in email.split("@")[1]:
        print(f"❌ Error: Invalid email format: {email}")
        return False
    
    # Validate password strength
    password_check = validate_password_strength(password)
    if not password_check["is_valid"]:
        print(f"❌ Error: Password does not meet security requirements:")
        for error in password_check["errors"]:
            print(f"   - {error}")
        print(f"\nPassword strength: {password_check['strength']}")
        return False
    
    try:
        db = SessionLocal()
        try:
            # Check if super admin already exists
            result = db.execute(
                select(AdminUser).where(
                    AdminUser.role == AdminRole.SUPER_ADMIN
                )
            )
            existing_super_admin = result.scalars().first()
            
            if existing_super_admin:
                print(f"✅ Super admin already exists: {existing_super_admin.email}")
                print("   Use this account to create additional admin users through the admin panel.")
                return True
            
            # Check if this email already exists
            result = db.execute(
                select(AdminUser).where(AdminUser.email == email.lower())
            )
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                print(f"❌ Error: Admin user with email {email} already exists")
                return False
            
            # Create the super admin user
            super_admin = AdminUser(
                name=name,
                email=email.lower(),
                role=AdminRole.SUPER_ADMIN,
                is_active=True,
                created_by=None  # Self-created
            )
            
            # Set password
            super_admin.set_password(password)
            
            # Add to database
            db.add(super_admin)
            db.commit()
            
            # Refresh to get the ID
            db.refresh(super_admin)
            
            print("✅ Super admin user created successfully!")
            print(f"   ID: {super_admin.id}")
            print(f"   Name: {super_admin.name}")
            print(f"   Email: {super_admin.email}")
            print(f"   Role: {super_admin.role.value}")
            print(f"   Created: {super_admin.created_at}")
            print(f"   Password strength: {password_check['strength']}")
            print("\n🚀 You can now log in to the admin panel with these credentials!")
            
            return True
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to create super admin: {e}")
        print(f"❌ Error creating super admin: {e}")
        return False

def create_sample_admin_users():
    """Create sample admin users for development/testing."""
    
    sample_users = [
        {
            "name": "Support Agent",
            "email": "support@example.com", 
            "role": AdminRole.SUPPORT_AGENT,
            "password": "SupportAgent123!"
        },
        {
            "name": "Data Analyst",
            "email": "analyst@example.com",
            "role": AdminRole.ANALYST, 
            "password": "DataAnalyst123!"
        },
        {
            "name": "Read Only User",
            "email": "readonly@example.com",
            "role": AdminRole.READ_ONLY,
            "password": "ReadOnlyUser123!"
        }
    ]
    
    try:
        db = SessionLocal()
        try:
            # Get super admin to set as creator
            result = db.execute(
                select(AdminUser).where(AdminUser.role == AdminRole.SUPER_ADMIN)
            )
            super_admin = result.scalars().first()
            
            if not super_admin:
                print("❌ No super admin found. Create super admin first.")
                return False
            
            created_count = 0
            
            for user_data in sample_users:
                # Check if user already exists
                result = db.execute(
                    select(AdminUser).where(AdminUser.email == user_data["email"])
                )
                existing_user = result.scalar_one_or_none()
                
                if existing_user:
                    print(f"⚠️  User {user_data['email']} already exists, skipping...")
                    continue
                
                # Create user
                user = AdminUser(
                    name=user_data["name"],
                    email=user_data["email"],
                    role=user_data["role"],
                    is_active=True,
                    created_by=super_admin.id
                )
                
                user.set_password(user_data["password"])
                
                db.add(user)
                created_count += 1
                
                print(f"✅ Created {user_data['role'].value}: {user_data['email']}")
            
            if created_count > 0:
                db.commit()
                print(f"\n🎉 Created {created_count} sample admin users!")
            else:
                print("ℹ️  No new sample users were created (all already exist)")
            
            return True
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to create sample users: {e}")
        print(f"❌ Error creating sample users: {e}")
        return False

def check_admin_panel_status():
    """Check the current status of the admin panel setup."""
    
    try:
        db = SessionLocal()
        try:
            # Count admin users by role
            result = db.execute(select(AdminUser))
            all_admins = result.scalars().all()
            
            if not all_admins:
                print("📊 Admin Panel Status: Not initialized")
                print("   No admin users exist. Run seed script to create the first super admin.")
                return
            
            # Count by role
            role_counts = {}
            active_count = 0
            
            for admin in all_admins:
                role_key = admin.role.value
                role_counts[role_key] = role_counts.get(role_key, 0) + 1
                if admin.is_active:
                    active_count += 1
            
            print("📊 Admin Panel Status: Initialized")
            print(f"   Total admin users: {len(all_admins)}")
            print(f"   Active admin users: {active_count}")
            print("   Users by role:")
            
            for role in AdminRole:
                count = role_counts.get(role.value, 0)
                print(f"     - {role.value}: {count}")
            
            print("\n👤 Admin Users:")
            for admin in sorted(all_admins, key=lambda x: x.created_at):
                status = "🟢 Active" if admin.is_active else "🔴 Inactive"
                print(f"   {status} | {admin.role.value:15} | {admin.email:30} | {admin.name}")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to check admin panel status: {e}")
        print(f"❌ Error checking admin panel status: {e}")

def print_usage():
    """Print usage instructions."""
    print("Admin Panel Seed Script")
    print("======================")
    print()
    print("Commands:")
    print("  python scripts/seed_admin.py init          - Create first super admin from env vars")
    print("  python scripts/seed_admin.py samples       - Create sample admin users for testing")
    print("  python scripts/seed_admin.py status        - Check current admin panel status")
    print("  python scripts/seed_admin.py help          - Show this help message")
    print()
    print("Environment Variables (for 'init' command):")
    print("  FIRST_SUPERADMIN_EMAIL      - Email for the first super admin (required)")
    print("  FIRST_SUPERADMIN_PASSWORD   - Password for the first super admin (required)")
    print("  FIRST_SUPERADMIN_NAME       - Name for the first super admin (optional)")
    print()
    print("Example:")
    print("  export FIRST_SUPERADMIN_EMAIL='admin@yourcompany.com'")
    print("  export FIRST_SUPERADMIN_PASSWORD='YourSecurePassword123!'")
    print("  export FIRST_SUPERADMIN_NAME='System Administrator'")
    print("  python scripts/seed_admin.py init")

def main():
    """Main entry point for the seed script."""
    
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    
    print("🔧 Eindr Admin Panel Seed Script")
    print("================================")
    print()
    
    if command == "init":
        print("Initializing admin panel with first super admin...")
        success = create_first_super_admin()
        if not success:
            sys.exit(1)
            
    elif command == "samples":
        print("Creating sample admin users...")
        success = create_sample_admin_users()
        if not success:
            sys.exit(1)
            
    elif command == "status":
        print("Checking admin panel status...")
        check_admin_panel_status()
        
    elif command == "help":
        print_usage()
        
    else:
        print(f"❌ Unknown command: {command}")
        print()
        print_usage()
        sys.exit(1)
    
    print()
    print("✨ Admin seed script completed!")

if __name__ == "__main__":
    main() 