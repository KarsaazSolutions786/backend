"""
Script to fix existing ledger entries with proper contact names
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
FIREBASE_TOKEN = "your_firebase_token_here"  # Replace with your actual token

def update_single_contact(ledger_id: str, contact_name: str):
    """Update a single ledger entry contact name."""
    
    url = f"{BASE_URL}/ledger/{ledger_id}/contact"
    headers = {
        "Authorization": f"Bearer {FIREBASE_TOKEN}",
        "Content-Type": "application/json"
    }
    
    params = {"contact_name": contact_name}
    
    try:
        response = requests.put(url, headers=headers, params=params)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Updated {ledger_id}: {result['data']['old_contact_name']} → {result['data']['new_contact_name']}")
            return True
        else:
            print(f"❌ Failed to update {ledger_id}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {ledger_id}: {e}")
        return False

def bulk_update_contacts(contact_mapping: dict):
    """Bulk update multiple ledger entries."""
    
    url = f"{BASE_URL}/ledger/bulk-update-contacts"
    headers = {
        "Authorization": f"Bearer {FIREBASE_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.put(url, headers=headers, json=contact_mapping)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Bulk update successful: {result['updated_count']} entries updated")
            for entry in result['updated_entries']:
                print(f"   {entry['ledger_id']}: {entry['old_contact_name']} → {entry['new_contact_name']}")
            return True
        else:
            print(f"❌ Bulk update failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error in bulk update: {e}")
        return False

def main():
    """Main function to fix contact names."""
    
    print("🔧 Ledger Contact Name Fixer")
    print("=" * 40)
    
    # Based on your database table, update these mappings
    # Format: "ledger_id": "actual_contact_name"
    
    contact_fixes = {
        "504b0705-1d5c-4ec0-a13b-47caf1bc3e1b": "John",      # $10000 entry
        "96172afd-ad6f-4211-a19f-9854beb4975b": "Sarah",     # $10000 entry  
        "9df2b48c-ae1a-4229-b365-af8e56fe054f": "Mike",      # $50 entry
        "b65fc313-075b-461a-ae53-353eaee54d1f": "David"      # $10000 entry
    }
    
    print("🎯 Contact name mappings:")
    for ledger_id, contact_name in contact_fixes.items():
        print(f"   {ledger_id[:8]}... → {contact_name}")
    
    print("\n🔄 Starting bulk update...")
    
    # Method 1: Bulk update (recommended)
    success = bulk_update_contacts(contact_fixes)
    
    if not success:
        print("\n🔄 Bulk update failed, trying individual updates...")
        
        # Method 2: Individual updates (fallback)
        for ledger_id, contact_name in contact_fixes.items():
            update_single_contact(ledger_id, contact_name)
    
    print("\n✅ Contact name fixing completed!")
    
    # Instructions for future entries
    print("\n📋 TO PREVENT THIS ISSUE IN FUTURE:")
    print("1. 🎤 Speak complete sentences: 'John owes me fifty dollars'")
    print("2. 🔊 Use clear, slow speech with good audio quality")
    print("3. 🎧 Use a good microphone in quiet environment")
    print("4. ⏱️  Make sure audio captures the full sentence")

if __name__ == "__main__":
    main() 