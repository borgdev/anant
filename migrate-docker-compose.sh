#!/bin/bash

# Anant Docker Compose Migration Script
# Migrates from multiple Docker Compose files to unified configuration

set -e

echo "🚀 Anant Docker Compose Unification Migration"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running from correct directory
if [[ ! -f "docker-compose.yml" ]]; then
    echo -e "${RED}❌ Error: docker-compose.yml not found. Run this script from the anant root directory.${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Current Docker Compose files found:${NC}"
find . -maxdepth 1 -name "docker-compose*.yml" -type f | while read file; do
    echo "  📄 $file"
done

echo ""
echo -e "${YELLOW}🔄 This script will:${NC}"
echo "  1. Backup existing Docker Compose files"
echo "  2. Create a unified docker-compose.yml with profiles"
echo "  3. Remove old separate configuration files"
echo "  4. Provide migration examples"

echo ""
read -p "Continue with migration? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⚠️  Migration cancelled by user${NC}"
    exit 0
fi

# Create backup directory
BACKUP_DIR="docker-compose-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo -e "${BLUE}📦 Creating backup in $BACKUP_DIR...${NC}"

# Backup existing files
for file in docker-compose*.yml; do
    if [[ -f "$file" && "$file" != "docker-compose.yml" ]]; then
        cp "$file" "$BACKUP_DIR/"
        echo "  ✅ Backed up: $file"
    fi
done

# Stop any running containers from old configurations
echo -e "${BLUE}🛑 Stopping existing containers...${NC}"
if [[ -f "docker-compose.dev.yml" ]]; then
    echo "  Stopping development services..."
    docker-compose -f docker-compose.dev.yml down --remove-orphans 2>/dev/null || true
fi

if [[ -f "docker-compose.registry.yml" ]]; then
    echo "  Stopping registry services..."
    docker-compose -f docker-compose.registry.yml down --remove-orphans 2>/dev/null || true
fi

# Remove old files
echo -e "${BLUE}🗑️  Removing old Docker Compose files...${NC}"
for file in docker-compose.dev.yml docker-compose.registry.yml; do
    if [[ -f "$file" ]]; then
        rm "$file"
        echo "  ✅ Removed: $file"
    fi
done

# Create migration examples
cat > "MIGRATION_EXAMPLES.md" << 'EOF'
# Docker Compose Migration Examples

## Old vs New Commands

### Development Environment
```bash
# OLD: Multiple files
docker-compose -f docker-compose.dev.yml up -d
docker-compose -f docker-compose.dev.yml down

# NEW: Unified with profiles
docker-compose --profile development up -d
docker-compose --profile development down
```

### Registry Environment
```bash
# OLD: Separate registry file
docker-compose -f docker-compose.registry.yml up -d
docker-compose -f docker-compose.registry.yml logs registry-api

# NEW: Registry profile
docker-compose --profile registry up -d
docker-compose logs anant-registry-api
```

### Production Environment
```bash
# OLD: Default file (production)
docker-compose up -d
docker-compose logs anant-ray-head

# NEW: Production profile (same services, clearer intent)
docker-compose --profile production up -d
docker-compose logs anant-ray-head
```

### Combined Environments
```bash
# NEW: Multiple profiles (not possible with old approach)
docker-compose --profile development --profile monitoring up -d
docker-compose --profile registry --profile jupyter up -d
docker-compose --profile production --profile monitoring up -d
```

## Service Name Mapping

| Old Service Name | New Service Name | Profile |
|------------------|------------------|---------|
| `anant-ray-head-dev` | `anant-ray-head-dev` | development |
| `anant-ray-worker-dev` | `anant-ray-worker-dev` | development |
| `anant-postgres-dev` | `anant-postgres-dev` | development |
| `anant-redis-dev` | `anant-cache` | development |
| `anant-jupyter` | `anant-jupyter` | jupyter |
| `postgres` (registry) | `anant-registry-db` | registry |
| `registry-api` | `anant-registry-api` | registry |
| `redis` (registry) | `anant-cache` | registry |

## Quick Migration Steps

1. **Update your scripts/CI/CD:**
   ```bash
   # Replace old commands with profile-based commands
   sed -i 's/docker-compose -f docker-compose.dev.yml/docker-compose --profile development/g' your-scripts.sh
   sed -i 's/docker-compose -f docker-compose.registry.yml/docker-compose --profile registry/g' your-scripts.sh
   ```

2. **Test the migration:**
   ```bash
   # Test development environment
   docker-compose --profile development up -d
   docker-compose --profile development ps
   docker-compose --profile development down
   
   # Test registry environment  
   docker-compose --profile registry up -d
   curl http://localhost:8080/health
   docker-compose --profile registry down
   ```

3. **Update documentation:**
   - Replace references to separate Docker Compose files
   - Update deployment instructions to use profiles
   - Update service names in monitoring/alerting configurations
EOF

echo -e "${GREEN}✅ Created migration examples: MIGRATION_EXAMPLES.md${NC}"

# Test the new configuration
echo -e "${BLUE}🧪 Testing unified configuration...${NC}"

# Validate Docker Compose syntax
if docker-compose config > /dev/null 2>&1; then
    echo -e "${GREEN}  ✅ Docker Compose syntax is valid${NC}"
else
    echo -e "${RED}  ❌ Docker Compose syntax error detected${NC}"
    echo "  Run 'docker-compose config' to see details"
fi

# List available profiles
echo -e "${BLUE}📋 Available profiles in unified configuration:${NC}"
echo "  🚀 production   - Full enterprise Ray cluster"
echo "  🛠️  development  - Lightweight dev environment" 
echo "  📊 registry     - Graph registry + Parquet storage"
echo "  📈 monitoring   - Prometheus + Grafana"
echo "  🔬 jupyter      - Interactive development"

echo ""
echo -e "${GREEN}🎉 Migration completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "  1. Review the unified docker-compose.yml file"
echo "  2. Read DOCKER_COMPOSE_UNIFIED.md for detailed usage"
echo "  3. Test your preferred deployment profile:"
echo "     ${BLUE}docker-compose --profile development up -d${NC}"
echo "  4. Update your deployment scripts to use profiles"
echo "  5. Check MIGRATION_EXAMPLES.md for command mappings"
echo ""
echo -e "${YELLOW}📦 Backup location:${NC} $BACKUP_DIR"
echo -e "${YELLOW}📚 Documentation:${NC} DOCKER_COMPOSE_UNIFIED.md"
echo -e "${YELLOW}🔄 Migration guide:${NC} MIGRATION_EXAMPLES.md"

# Optional: Test a profile
echo ""
read -p "Would you like to test the development profile? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🧪 Testing development profile...${NC}"
    docker-compose --profile development up -d
    sleep 5
    echo ""
    echo -e "${BLUE}📊 Service status:${NC}"
    docker-compose ps
    echo ""
    echo -e "${YELLOW}Services are running. Access points:${NC}"
    echo "  Ray Dashboard: http://localhost:8266"
    echo "  Anant API: http://localhost:8001" 
    echo ""
    echo "Run 'docker-compose --profile development down' to stop services"
fi

echo ""
echo -e "${GREEN}✨ Docker Compose unification complete!${NC}"