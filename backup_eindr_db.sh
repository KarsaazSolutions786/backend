#!/bin/bash

# Eindr Database Backup Script
# ============================

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🗄️  Eindr Database Backup Script${NC}"
echo "=================================="

# Create backups directory if it doesn't exist
mkdir -p ./backups

# Generate timestamp for unique backup name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="./backups/eindr_db_backup_${TIMESTAMP}.sql"

echo -e "${BLUE}📊 Starting backup...${NC}"

# Method 1: Direct backup (recommended)
if command -v pg_dump &> /dev/null; then
    echo "Using pg_dump (direct method)..."
    pg_dump -h localhost -p 5433 -U eindr -d eindr_db -f "${BACKUP_FILE}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Backup created successfully!${NC}"
        echo -e "${GREEN}📁 File: ${BACKUP_FILE}${NC}"
        echo -e "${GREEN}📏 Size: $(du -h "${BACKUP_FILE}" | cut -f1)${NC}"
    else
        echo -e "${RED}❌ Backup failed!${NC}"
        exit 1
    fi
else
    # Method 2: Using Docker exec
    echo "Using Docker exec method..."
    docker exec backend-new-postgres-server-1 pg_dump -U eindr -d eindr_db -f /tmp/eindr_backup_temp.sql
    docker cp backend-new-postgres-server-1:/tmp/eindr_backup_temp.sql "${BACKUP_FILE}"
    docker exec backend-new-postgres-server-1 rm /tmp/eindr_backup_temp.sql
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Backup created successfully!${NC}"
        echo -e "${GREEN}📁 File: ${BACKUP_FILE}${NC}"
        echo -e "${GREEN}📏 Size: $(du -h "${BACKUP_FILE}" | cut -f1)${NC}"
    else
        echo -e "${RED}❌ Backup failed!${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}📋 All backups in ./backups/:${NC}"
ls -la ./backups/

echo ""
echo -e "${GREEN}🎉 Backup completed successfully!${NC}" 