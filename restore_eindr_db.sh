#!/bin/bash

# Eindr Database Restore Script
# =============================

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Eindr Database Restore Script${NC}"
echo "=================================="

# Check if backup file is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}📋 Available backup files:${NC}"
    ls -la ./backups/*.sql 2>/dev/null || echo "No backup files found in ./backups/"
    echo ""
    echo -e "${RED}❌ Usage: $0 <backup_file.sql>${NC}"
    echo -e "${BLUE}💡 Example: $0 ./backups/eindr_db_backup_20250707_111241.sql${NC}"
    exit 1
fi

BACKUP_FILE="$1"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}❌ Backup file not found: $BACKUP_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}📁 Backup file: $BACKUP_FILE${NC}"
echo -e "${BLUE}📏 File size: $(du -h "$BACKUP_FILE" | cut -f1)${NC}"

# Warning about overwriting data
echo ""
echo -e "${YELLOW}⚠️  WARNING: This will overwrite all data in eindr_db!${NC}"
echo -e "${YELLOW}⚠️  Make sure you have a current backup before proceeding.${NC}"
echo ""
read -p "Are you sure you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${BLUE}🚫 Restore cancelled.${NC}"
    exit 0
fi

echo -e "${BLUE}🔄 Starting restore...${NC}"

# Method 1: Direct restore (recommended)
if command -v psql &> /dev/null; then
    echo "Using psql (direct method)..."
    
    # Drop and recreate database
    echo "Recreating database..."
    psql -h localhost -p 5433 -U eindr -d postgres -c "DROP DATABASE IF EXISTS eindr_db;" 
    psql -h localhost -p 5433 -U eindr -d postgres -c "CREATE DATABASE eindr_db OWNER eindr;"
    
    # Restore from backup
    echo "Restoring from backup..."
    psql -h localhost -p 5433 -U eindr -d eindr_db -f "$BACKUP_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Database restored successfully!${NC}"
    else
        echo -e "${RED}❌ Restore failed!${NC}"
        exit 1
    fi
else
    # Method 2: Using Docker exec
    echo "Using Docker exec method..."
    
    # Copy backup file to container
    docker cp "$BACKUP_FILE" backend-new-postgres-server-1:/tmp/restore_backup.sql
    
    # Drop and recreate database
    echo "Recreating database..."
    docker exec backend-new-postgres-server-1 psql -U eindr -d postgres -c "DROP DATABASE IF EXISTS eindr_db;"
    docker exec backend-new-postgres-server-1 psql -U eindr -d postgres -c "CREATE DATABASE eindr_db OWNER eindr;"
    
    # Restore from backup
    echo "Restoring from backup..."
    docker exec backend-new-postgres-server-1 psql -U eindr -d eindr_db -f /tmp/restore_backup.sql
    
    # Cleanup
    docker exec backend-new-postgres-server-1 rm /tmp/restore_backup.sql
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Database restored successfully!${NC}"
    else
        echo -e "${RED}❌ Restore failed!${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}🎉 Restore completed successfully!${NC}"
echo -e "${BLUE}💡 You may need to restart your microservices to pick up the restored data.${NC}" 