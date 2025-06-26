.PHONY: help build up down logs clean migrate test lint format

# Colors for help
YELLOW := \033[33m
RESET := \033[0m

# Default target
help: ## Show this help message
	@echo "Eindr Microservices Development Commands"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}'

# =======================
# DEVELOPMENT COMMANDS
# =======================

build: ## Build all microservices
	docker-compose -f docker-compose.microservices.yml build

up: ## Start all services in development mode
	docker-compose -f docker-compose.microservices.yml up -d
	@echo "🚀 All services started!"
	@echo "API Gateway: http://localhost:8080"
	@echo "RabbitMQ UI: http://localhost:15672 (eindr/eindr123)"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

down: ## Stop all services
	docker-compose -f docker-compose.microservices.yml down

logs: ## Follow logs for all services
	docker-compose -f docker-compose.microservices.yml logs -f

clean: ## Remove all containers, volumes, and images
	docker-compose -f docker-compose.microservices.yml down -v --remove-orphans
	docker system prune -f

# =======================
# DATABASE MANAGEMENT
# =======================

setup-databases: ## Create all databases and tables locally
	@echo "🏗️  Setting up all microservice databases..."
	./run_database_setup.sh

run-migrations: ## Run database migrations locally
	@echo "🔄 Running all database migrations..."
	./run_migrations.sh

test-databases: ## Test database connections
	@echo "🧪 Testing database connections..."
	./test_databases.sh

init-databases: setup-databases run-migrations ## Initialize databases (create + migrate)

# Local PostgreSQL with Docker
start-local-db: ## Start local PostgreSQL in Docker
	@echo "🐘 Starting local PostgreSQL database..."
	docker-compose -f docker-compose.local-db.yml up -d postgres
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 10
	@echo "✅ PostgreSQL is running on localhost:5432"
	@echo "📊 Connection details:"
	@echo "   Host: localhost:5432"
	@echo "   User: eindr_user"
	@echo "   Password: eindr_pass"
	@echo "   Databases: All microservice databases created automatically"

stop-local-db: ## Stop local PostgreSQL
	@echo "🛑 Stopping local PostgreSQL database..."
	docker-compose -f docker-compose.local-db.yml down

start-pgadmin: ## Start PgAdmin for database management
	@echo "🌐 Starting PgAdmin..."
	docker-compose -f docker-compose.local-db.yml up -d pgadmin
	@echo "✅ PgAdmin is running at http://localhost:5050"
	@echo "📧 Email: admin@eindr.local"
	@echo "🔑 Password: admin123"

local-db-logs: ## View PostgreSQL logs
	docker-compose -f docker-compose.local-db.yml logs -f postgres

test-local-db: ## Test local database connection
	@echo "🧪 Testing local PostgreSQL connection..."
	@docker exec -it $$(docker-compose -f docker-compose.local-db.yml ps -q postgres) psql -U eindr_user -d eindr_dev -c "SELECT 'PostgreSQL is working!' as status;"

reset-local-db: stop-local-db ## Reset local database (destroys all data)
	docker-compose -f docker-compose.local-db.yml down -v
	docker-compose -f docker-compose.local-db.yml up -d postgres

setup-local-env: ## Set up environment variables for local development
	@echo "🔧 To set up your local environment, run:"
	@echo "   source setup_local_env.sh"
	@echo ""
	@echo "💡 This will configure all DATABASE_URLs to point to your local PostgreSQL"

dev-ready: start-local-db setup-local-env ## Get everything ready for local development
	@echo ""
	@echo "🎯 Local development environment is ready!"
	@echo "📋 Quick summary:"
	@echo "   1. PostgreSQL: localhost:5432 (eindr_user/eindr_pass)"
	@echo "   2. All 10 databases created and ready"
	@echo "   3. PgAdmin: http://localhost:5050 (admin@eindr.dev/admin123)"
	@echo ""
	@echo "▶️  Next steps:"
	@echo "   • Run: source setup_local_env.sh"
	@echo "   • Start your microservices individually"
	@echo "   • Or use: make up (for full Docker deployment)"

migrate-all: migrate-auth migrate-user migrate-reminder migrate-note migrate-ledger migrate-friend migrate-history migrate-chat migrate-scheduler ## Run all migrations

migrate-auth: ## Run migrations for auth service
	docker-compose -f docker-compose.microservices.yml exec auth-service alembic upgrade head

migrate-user: ## Run migrations for user service
	docker-compose -f docker-compose.microservices.yml exec user-service alembic upgrade head

migrate-reminder: ## Run migrations for reminder service
	docker-compose -f docker-compose.microservices.yml exec reminder-service alembic upgrade head

migrate-note: ## Run migrations for note service
	docker-compose -f docker-compose.microservices.yml exec note-service alembic upgrade head

migrate-ledger: ## Run migrations for ledger service
	docker-compose -f docker-compose.microservices.yml exec ledger-service alembic upgrade head

migrate-friend: ## Run migrations for friend service
	docker-compose -f docker-compose.microservices.yml exec friend-service alembic upgrade head

migrate-history: ## Run migrations for history service
	docker-compose -f docker-compose.microservices.yml exec history-service alembic upgrade head

migrate-chat: ## Run migrations for chat service
	docker-compose -f docker-compose.microservices.yml exec chat-service alembic upgrade head

migrate-scheduler: ## Run migrations for scheduler service
	docker-compose -f docker-compose.microservices.yml exec scheduler-service alembic upgrade head

# =======================
# TESTING & QUALITY
# =======================

test: ## Run tests for all services
	@echo "Running tests for all services..."
	docker-compose -f docker-compose.microservices.yml exec auth-service pytest tests/
	docker-compose -f docker-compose.microservices.yml exec user-service pytest tests/

lint: ## Run linting for all services
	find services/ -name "*.py" -exec flake8 {} \;

format: ## Format code for all services
	find services/ -name "*.py" -exec black {} \;

# =======================
# KONG API GATEWAY
# =======================

kong-setup: ## Setup Kong API Gateway with all services
	@echo "🔧 Setting up Kong API Gateway..."
	./services/api-gateway/kong-setup.sh

kong-logs: ## Follow Kong logs
	docker-compose -f docker-compose.microservices.yml logs -f kong

kong-restart: ## Restart Kong gateway
	docker-compose -f docker-compose.microservices.yml restart kong

kong-admin: ## Show Kong admin URLs
	@echo "🌐 Kong Admin API: http://localhost:8101"
	@echo "📊 Kong Admin GUI: http://localhost:8102"
	@echo "🖥️  Konga UI: http://localhost:8103"

konga-logs: ## Follow Konga admin UI logs
	docker-compose -f docker-compose.microservices.yml logs -f konga

# =======================
# UTILITIES
# =======================

health-check: ## Check health of all services
	@echo "Checking service health..."
	@curl -f http://localhost:8080/health || echo "❌ API Gateway down"
	@curl -f http://localhost:8080/auth/health || echo "❌ Auth service down"
	@curl -f http://localhost:8080/users/health || echo "❌ User service down"
	@curl -f http://localhost:8080/reminders/health || echo "❌ Reminder service down"
	@echo "✅ Health check complete"

reset: clean build up migrate-all kong-setup ## Reset entire environment 