# mem0-mcp-selfhosted — setup targets
# Usage:
#   make install            # Qdrant vector store (default, no system deps)
#   make install-pgvector   # pgvector vector store (installs system libs)
#   make start              # Start all services for current vector store
#   make stop               # Stop all running services
#   make status             # Show service health

OS     := $(shell uname -s)
ARCH   := $(shell uname -m)
PYTHON := python3

QDRANT_COMPOSE     := $(CURDIR)/../../docker-compose.yml
PGVECTOR_COMPOSE   := $(CURDIR)/../../docker-compose.pgvector.yml

.PHONY: install install-pgvector start stop status \
        _check-docker _check-uv _install-ollama _check-ollama \
        _install-pg-dev _start-qdrant _start-pgvector \
        _install-mem0-qdrant _install-mem0-pgvector

# ─── Default: Qdrant ──────────────────────────────────────────────────────────

install: _check-docker _check-uv _check-ollama _install-mem0-qdrant _start-qdrant
	@echo ""
	@echo "✅  mem0 (Qdrant) ready."
	@echo ""
	@echo "Register the MCP server (run once):"
	@echo ""
	@echo "  claude mcp add --scope user --transport stdio mem0 \\"
	@echo "    --env MEM0_USER_ID=<your-user-id> \\"
	@echo "    --env MEM0_AUTHOR_ID=<your-username> \\"
	@echo "    --env MEM0_TEAM_ID=<your-team> \\"
	@echo "    --env MEM0_EMBED_PROVIDER=ollama \\"
	@echo "    --env MEM0_EMBED_MODEL=nomic-embed-text \\"
	@echo "    --env MEM0_EMBED_DIMS=768 \\"
	@echo "    --env MEM0_QDRANT_URL=http://localhost:6333 \\"
	@echo "    --env MEM0_OLLAMA_URL=http://localhost:11434 \\"
	@echo "    --env MEM0_PROVIDER=anthropic \\"
	@echo "    --env MEM0_TELEMETRY=false \\"
	@echo "    -- uvx --from git+https://github.com/justinb-dfw/mem0-mcp-selfhosted.git mem0-mcp-selfhosted"

# ─── pgvector path ────────────────────────────────────────────────────────────

install-pgvector: _check-docker _check-uv _check-ollama _install-pg-dev _install-mem0-pgvector _start-pgvector
	@echo ""
	@echo "✅  mem0 (pgvector + Neo4j) ready."
	@echo ""
	@echo "Local Docker services started. If your team has a shared pgvector or Neo4j"
	@echo "instance, replace MEM0_PG_HOST / MEM0_NEO4J_URL with those endpoints instead."
	@echo ""
	@echo "Register the MCP server (run once):"
	@echo ""
	@echo "  claude mcp add --scope user --transport stdio mem0 \\"
	@echo "    --env MEM0_USER_ID=<your-user-id> \\"
	@echo "    --env MEM0_AUTHOR_ID=<your-username> \\"
	@echo "    --env MEM0_TEAM_ID=<your-team> \\"
	@echo "    --env MEM0_EMBED_PROVIDER=ollama \\"
	@echo "    --env MEM0_EMBED_MODEL=nomic-embed-text \\"
	@echo "    --env MEM0_EMBED_DIMS=768 \\"
	@echo "    --env MEM0_VECTOR_STORE=pgvector \\"
	@echo "    --env MEM0_PG_HOST=localhost \\"
	@echo "    --env MEM0_PG_PORT=5432 \\"
	@echo "    --env MEM0_PG_USER=mem0 \\"
	@echo "    --env MEM0_PG_PASSWORD=mem0 \\"
	@echo "    --env MEM0_PG_DB=mem0 \\"
	@echo "    --env MEM0_OLLAMA_URL=http://localhost:11434 \\"
	@echo "    --env MEM0_PROVIDER=anthropic \\"
	@echo "    --env MEM0_ENABLE_GRAPH=true \\"
	@echo "    --env MEM0_NEO4J_URL=bolt://localhost:7687 \\"
	@echo "    --env MEM0_NEO4J_USER=neo4j \\"
	@echo "    --env MEM0_NEO4J_PASSWORD=mem0graph \\"
	@echo "    --env MEM0_TELEMETRY=false \\"
	@echo "    -- uvx --from git+https://github.com/justinb-dfw/mem0-mcp-selfhosted.git mem0-mcp-selfhosted"
	@echo ""
	@echo "Neo4j Browser UI: http://localhost:7474  (user: neo4j / pass: mem0graph)"

# ─── Service management ───────────────────────────────────────────────────────

start: _check-docker _check-ollama
	@if docker ps --format '{{.Names}}' | grep -q '^mem0-pgvector$$'; then \
		echo "pgvector is running — starting pgvector stack"; \
		docker compose -f $(PGVECTOR_COMPOSE) up -d; \
	else \
		echo "Starting Qdrant stack"; \
		docker compose -f $(QDRANT_COMPOSE) up -d; \
	fi

stop:
	@docker compose -f $(QDRANT_COMPOSE) down 2>/dev/null || true
	@docker compose -f $(PGVECTOR_COMPOSE) down 2>/dev/null || true
	@echo "Services stopped."

status:
	@echo "=== Qdrant ==="
	@curl -sf http://localhost:6333/healthz 2>/dev/null && echo "✅  healthy" || echo "❌  not running"
	@echo "=== pgvector (PostgreSQL) ==="
	@docker exec mem0-postgres pg_isready -U mem0 2>/dev/null && echo "✅  healthy" || echo "❌  not running"
	@echo "=== Ollama ==="
	@curl -sf http://localhost:11434/api/tags 2>/dev/null | $(PYTHON) -c "import sys,json; d=json.load(sys.stdin); print('✅  ' + str(len(d['models'])) + ' model(s) loaded')" 2>/dev/null || echo "❌  not running"
	@echo "=== Neo4j ==="
	@curl -sf http://localhost:7474 >/dev/null 2>&1 && echo "✅  healthy (browser: http://localhost:7474)" || echo "❌  not running"
	@echo "=== nomic-embed-text ==="
	@ollama list 2>/dev/null | grep -q nomic-embed-text && echo "✅  available" || echo "❌  not pulled (run: ollama pull nomic-embed-text)"

# ─── mem0 Python package install ─────────────────────────────────────────────

# Qdrant path: installs base mem0ai[llms] (no pg_config needed)
_install-mem0-qdrant:
	@echo "Installing mem0 Python dependencies (Qdrant path)..."
	@pip3 install --break-system-packages -q mem0ai[llms] qdrant-client ollama anthropic pyyaml python-dotenv 2>&1 | tail -3
	@echo "✅  mem0 Python deps installed"

# pgvector path: pg_config must already be on PATH (run after _install-pg-dev)
# Also installs psycopg2 (now builds correctly) and pgvector Python client
_install-mem0-pgvector:
	@echo "Installing mem0 Python dependencies (pgvector path)..."
	@pip3 install --break-system-packages -q "mem0ai[graph,llms]" pgvector psycopg2 ollama anthropic pyyaml python-dotenv 2>&1 | tail -3
	@echo "✅  mem0 Python deps installed (with pgvector + psycopg2)"

# ─── Internal helpers ─────────────────────────────────────────────────────────

_check-docker:
	@command -v docker >/dev/null 2>&1 || (echo "❌  Docker not found. Install Docker Desktop: https://www.docker.com/products/docker-desktop/" && exit 1)
	@docker info >/dev/null 2>&1 || (echo "❌  Docker is not running. Start Docker Desktop and retry." && exit 1)
	@echo "✅  Docker is running"

_check-uv:
ifeq ($(OS),Darwin)
	@command -v uv >/dev/null 2>&1 || brew install uv
else
	@command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
endif
	@echo "✅  uv $(shell uv --version 2>/dev/null)"

_install-ollama:
ifeq ($(OS),Darwin)
	@brew install ollama
else
	@curl -fsSL https://ollama.com/install.sh | sh
endif

_check-ollama:
	@command -v ollama >/dev/null 2>&1 || $(MAKE) _install-ollama
	@pgrep -x ollama >/dev/null 2>&1 || (ollama serve >/dev/null 2>&1 & sleep 2)
	@ollama list 2>/dev/null | grep -q nomic-embed-text || (echo "Pulling nomic-embed-text embedding model (~274MB)..." && ollama pull nomic-embed-text)
	@echo "✅  Ollama ready with nomic-embed-text"
	@echo ""
	@echo "⚠️   NOTE: The MCP server uses MEM0_OLLAMA_URL (default: http://localhost:11434)."
	@echo "    If your Ollama is remote, in Docker, or on a different port, set"
	@echo "    --env MEM0_OLLAMA_URL=http://<your-ollama-host>:<port> in the claude mcp add command."

_install-pg-dev:
	@echo "Installing PostgreSQL development headers (required for psycopg2)..."
ifeq ($(OS),Darwin)
	@command -v pg_config >/dev/null 2>&1 || brew install postgresql@16
	@echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$$PATH"' >> ~/.zshrc 2>/dev/null || true
	@echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$$PATH"' >> ~/.bash_profile 2>/dev/null || true
	$(eval export PATH := /opt/homebrew/opt/postgresql@16/bin:$(PATH))
else ifeq ($(OS),Linux)
	@command -v pg_config >/dev/null 2>&1 || ( \
		if command -v apt-get >/dev/null 2>&1; then \
			apt-get update -qq && apt-get install -y libpq-dev postgresql-client; \
		elif command -v dnf >/dev/null 2>&1; then \
			dnf install -y postgresql-devel; \
		elif command -v yum >/dev/null 2>&1; then \
			yum install -y postgresql-devel; \
		else \
			echo "❌  Unknown Linux distro. Install libpq-dev / postgresql-devel manually." && exit 1; \
		fi \
	)
endif
	@command -v pg_config >/dev/null 2>&1 && echo "✅  pg_config found at $$(command -v pg_config)" || (echo "❌  pg_config still not found. You may need to open a new shell." && exit 1)

_start-qdrant:
	@docker ps --format '{{.Names}}' | grep -q '^mem0-qdrant$$' && echo "✅  Qdrant already running" || \
		(echo "Starting Qdrant..." && docker compose -f $(QDRANT_COMPOSE) up -d)
	@for i in $$(seq 1 15); do \
		curl -sf http://localhost:6333/healthz >/dev/null 2>&1 && echo "✅  Qdrant healthy (port 6333)" && break; \
		[ "$$i" -eq 15 ] && echo "❌  Qdrant did not start in time" && exit 1; \
		sleep 2; \
	done

_start-pgvector:
	@docker ps --format '{{.Names}}' | grep -q '^mem0-postgres$$' && echo "✅  pgvector already running" || \
		(echo "Starting pgvector (PostgreSQL + pgvector extension)..." && docker compose -f $(PGVECTOR_COMPOSE) up -d)
	@for i in $$(seq 1 15); do \
		docker exec mem0-postgres pg_isready -U mem0 >/dev/null 2>&1 && echo "✅  pgvector healthy (port 5432)" && break; \
		[ "$$i" -eq 15 ] && echo "❌  pgvector did not start in time" && exit 1; \
		sleep 2; \
	done
	@docker exec mem0-postgres psql -U mem0 -d mem0 -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null 2>&1 && \
		echo "✅  pgvector extension enabled" || echo "⚠️   pgvector extension may already be enabled"
