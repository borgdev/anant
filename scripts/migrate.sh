#!/bin/bash
# Alembic migration script for development environment
# Usage: ./migrate.sh [command]
# Commands: 
#   current  - Show current migration
#   upgrade  - Apply all pending migrations
#   revision - Create new migration (requires message)

set -e

# Development database configuration
export POSTGRES_HOST=postgres-dev
export POSTGRES_USER=anant_dev
export POSTGRES_PASSWORD=dev_password
export POSTGRES_DB=anant_dev

# Default to upgrade if no command provided
COMMAND=${1:-upgrade}

echo "🔧 Running Alembic $COMMAND for development environment..."
echo "📊 Database: $POSTGRES_DB@$POSTGRES_HOST:5432"

case $COMMAND in
    "current")
        docker exec -w /app/anant_api anant-ray-head-dev \
            env POSTGRES_HOST=$POSTGRES_HOST POSTGRES_USER=$POSTGRES_USER \
                POSTGRES_PASSWORD=$POSTGRES_PASSWORD POSTGRES_DB=$POSTGRES_DB \
            alembic current
        ;;
    "upgrade")
        docker exec -w /app/anant_api anant-ray-head-dev \
            env POSTGRES_HOST=$POSTGRES_HOST POSTGRES_USER=$POSTGRES_USER \
                POSTGRES_PASSWORD=$POSTGRES_PASSWORD POSTGRES_DB=$POSTGRES_DB \
            alembic upgrade head
        ;;
    "revision")
        if [ -z "$2" ]; then
            echo "❌ Error: revision command requires a message"
            echo "Usage: ./migrate.sh revision 'Your migration message'"
            exit 1
        fi
        docker exec -w /app/anant_api anant-ray-head-dev \
            env POSTGRES_HOST=$POSTGRES_HOST POSTGRES_USER=$POSTGRES_USER \
                POSTGRES_PASSWORD=$POSTGRES_PASSWORD POSTGRES_DB=$POSTGRES_DB \
            alembic revision --autogenerate -m "$2"
        ;;
    "history")
        docker exec -w /app/anant_api anant-ray-head-dev \
            env POSTGRES_HOST=$POSTGRES_HOST POSTGRES_USER=$POSTGRES_USER \
                POSTGRES_PASSWORD=$POSTGRES_PASSWORD POSTGRES_DB=$POSTGRES_DB \
            alembic history
        ;;
    *)
        echo "❌ Unknown command: $COMMAND"
        echo "Available commands: current, upgrade, revision, history"
        exit 1
        ;;
esac

echo "✅ Alembic $COMMAND completed successfully"