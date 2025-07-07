#!/bin/bash

echo "🔄 Updating all microservices to use shared eindr_db database..."

# Backup the original file
cp docker-compose.microservices.yml docker-compose.microservices.yml.backup

# Update reminder-service
sed -i '' 's/postgresql:\/\/postgres:postgres@reminder-db:5432\/reminder_db/postgresql:\/\/eindr:eindr_pass@new-postgres-server:5432\/eindr_db/g' docker-compose.microservices.yml

# Update note-service
sed -i '' 's/postgresql:\/\/postgres:postgres@note-db:5432\/note_db/postgresql:\/\/eindr:eindr_pass@new-postgres-server:5432\/eindr_db/g' docker-compose.microservices.yml

# Update ledger-service
sed -i '' 's/postgresql:\/\/postgres:postgres@ledger-db:5432\/ledger_db/postgresql:\/\/eindr:eindr_pass@new-postgres-server:5432\/eindr_db/g' docker-compose.microservices.yml

# Update friend-service
sed -i '' 's/postgresql:\/\/postgres:postgres@friend-db:5432\/friend_db/postgresql:\/\/eindr:eindr_pass@new-postgres-server:5432\/eindr_db/g' docker-compose.microservices.yml

# Update history-service
sed -i '' 's/postgresql:\/\/postgres:postgres@history-db:5432\/history_db/postgresql:\/\/eindr:eindr_pass@new-postgres-server:5432\/eindr_db/g' docker-compose.microservices.yml

# Update chat-service
sed -i '' 's/postgresql:\/\/postgres:postgres@chat-db:5432\/chat_db/postgresql:\/\/eindr:eindr_pass@new-postgres-server:5432\/eindr_db/g' docker-compose.microservices.yml

# Update scheduler-service
sed -i '' 's/postgresql:\/\/postgres:postgres@scheduler-db:5432\/scheduler_db/postgresql:\/\/eindr:eindr_pass@new-postgres-server:5432\/eindr_db/g' docker-compose.microservices.yml

echo "✅ Database URLs updated in docker-compose.microservices.yml"
echo "📝 Backup saved as docker-compose.microservices.yml.backup" 