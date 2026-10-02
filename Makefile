.DEFAULT_GOAL := help
PY := backend/.venv/bin/python
PIP := backend/.venv/bin/pip
ifeq ($(OS),Windows_NT)
	PY := backend/.venv/Scripts/python.exe
	PIP := backend/.venv/Scripts/pip.exe
endif

.PHONY: help
help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- setup -----------------------------------------------------------------

.PHONY: setup
setup: setup-backend setup-frontend ## Install everything

.PHONY: setup-backend
setup-backend: ## Create the venv and install Python dependencies
	python -m venv backend/.venv
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt

.PHONY: setup-ai
setup-ai: ## Additionally install the optional torch stack
	$(PIP) install -r backend/requirements-ai.txt

.PHONY: setup-frontend
setup-frontend: ## Install Node dependencies
	cd frontend && npm install

# --- run -------------------------------------------------------------------

.PHONY: api
api: ## Run the API with reload on :8000
	cd backend && ../$(PY) -m uvicorn app.main:app --reload --port 8000

.PHONY: web
web: ## Run the dashboard on :3000
	cd frontend && npm run dev

.PHONY: up
up: ## Start the full stack in Docker
	docker compose up --build

.PHONY: down
down: ## Stop the stack
	docker compose down

# --- quality ---------------------------------------------------------------

.PHONY: test
test: ## Unit tests (no network)
	cd backend && ../$(PY) -m pytest -q

.PHONY: test-live
test-live: ## Integration tests against the live satellite archive
	cd backend && ../$(PY) -m pytest -q -m live

.PHONY: typecheck
typecheck: ## Typecheck the frontend
	cd frontend && npx tsc --noEmit

.PHONY: lint
lint: ## Lint the backend (requires ruff)
	cd backend && ../$(PY) -m ruff check app tests

.PHONY: check
check: test typecheck ## Everything that must pass before a commit

# --- utilities -------------------------------------------------------------

.PHONY: db-init
db-init: ## Create the PostGIS schema (needs ATLASAI_DATABASE_URL)
	cd backend && ../$(PY) -c "from app.db.session import init_schema; init_schema()"

.PHONY: smoke
smoke: ## One live analysis, end to end
	@echo "Al Massira reservoir, 2024 vs 2019 --"
	@curl -sS "http://localhost:8000/api/v1/analysis/water?bbox=-7.75,32.45,-7.50,32.62&start=2024-06-01&end=2024-08-31&baseline_start=2019-06-01&baseline_end=2019-08-31&resolution=60" \
		| $(PY) -c "import json,sys; d=json.load(sys.stdin); print(d['summary']); print('confidence:', d['confidence'])"

.PHONY: clean
clean: ## Remove caches and build output
	rm -rf backend/.pytest_cache backend/.cache frontend/.next
	find backend -name __pycache__ -type d -prune -exec rm -rf {} +
