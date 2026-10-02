# Deployment

The platform is I/O-bound, not compute-bound. Almost every second of a request
is spent waiting on HTTP range reads against remote cloud archives, which
determines nearly all of the operational advice below.

---

## Single instance (Docker Compose)

```bash
cp .env.example .env
docker compose up --build -d
docker compose logs -f api
```

Good for evaluation, demos, and a team-sized deployment. The in-process result
cache works correctly because there is exactly one process holding it.

---

## What to change before production

### 1. The result cache is process-local

`app/services/cache.py` holds computed rasters in memory, keyed by a hash of the
request. The layer-PNG endpoint reads from it.

**With more than one worker, this breaks in a specific and confusing way**: the
worker that ran the analysis holds the pixels, but a later request for
`/analysis/result/{key}/layer/{layer}.png` may land on a different worker and
404 with "result expired or unknown" even though the analysis just succeeded.

Options, in order of preference:

- **Keep one worker per container and scale containers**, with sticky sessions
  or a consistent-hash route on the result key. Simplest correct answer.
- **Move the cache to Redis.** Rasters are large, so store PNG bytes rather
  than NumPy arrays, and set `maxmemory-policy allkeys-lru`.
- **Render PNGs to object storage** at analysis time and return signed URLs.
  Best at scale — the API stops serving image bytes entirely.

### 2. Get a Planetary Computer subscription key

Free, and the single highest-leverage change for a multi-user deployment.
Anonymous access is rate-limited at the SAS token endpoint, which is the first
thing to fail under concurrency.

Sign in at <https://planetarycomputer.developer.azure-api.net/> and copy the key
from your profile — no payment details, and the data itself is free either way.

```bash
ATLASAI_PC_SUBSCRIPTION_KEY=...
```

### 3. Set a real pixel budget

`ATLASAI_MAX_PIXELS` (default 4,000,000) caps the grid per analysis; larger
areas are automatically coarsened rather than rejected. Lower it for a
latency-sensitive deployment, raise it if you have bandwidth and patience.

### 4. Bound request time

A large area at fine resolution over many scenes is a genuinely long request.
Put a timeout in front of the API and set `max_items` conservatively — each
additional scene is roughly five more windowed reads.

---

## Scaling

### Workers

The endpoints are synchronous and run in FastAPI's threadpool, so a single
container already overlaps many in-flight reads. Because the work is
network-bound rather than CPU-bound, more threads help more than more cores.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Scale by adding containers, not workers, until the cache is externalised.

### Colocate with the data

Planetary Computer assets live in Azure **West Europe**. Deploying there turns
cross-continent range reads into intra-region ones and is the largest single
latency win available — typically more than any code change.

### Background analysis

Long-running work (national scans, time series over a decade, scheduled
monitoring) does not belong in a request. Move it to a worker queue —
`AnalysisRun` and `IndexObservation` in `app/db/models.py` exist to hold the
results — and have the API read precomputed rows.

---

## Persistence

Optional. Without `ATLASAI_DATABASE_URL` the platform runs entirely from the
live archive.

Turn it on when you need:

- **Provenance** — which module, window, resolution and scene ids produced a
  published figure, months later.
- **Alerting** — threshold crossings detected by a scheduled job.
- **Official boundaries** — replacing the approximate envelopes in
  `app/services/regions.py`. **Do this before publishing area statistics.**
- **Cached time series** — a ten-year NDVI curve is ~40 imagery reads on every
  request otherwise.

```python
from app.db.session import init_schema
init_schema()   # enables PostGIS and creates tables; safe to re-run
```

Loading boundaries is documented in `infra/postgres/init.sql`.

---

## Kubernetes

Nothing exotic is required — the API is a stateless HTTP service once the cache
is externalised.

```yaml
resources:
  requests: { cpu: "500m", memory: "1Gi" }
  limits:   { cpu: "2",     memory: "4Gi" }
```

Memory is the constraint, not CPU: a 4-million-pixel analysis holding several
float32 bands plus intermediates runs to a few hundred MB, and concurrent
requests multiply that. Size limits against `ATLASAI_MAX_PIXELS`, and prefer
vertical headroom over horizontal replicas for the memory dimension.

Probes:

```yaml
readinessProbe:
  httpGet: { path: /api/v1/health, port: 8000 }
  initialDelaySeconds: 10
livenessProbe:
  httpGet: { path: /api/v1/health, port: 8000 }
  periodSeconds: 30
```

`/api/v1/health` reports subsystem availability and cache state without
touching the network, so it stays fast and does not fail because a satellite
archive is briefly unreachable.

### GPU

Only needed if you install trained weights (see [MODELS.md](MODELS.md)). Without
them the analytic implementations are pure NumPy and a GPU node is wasted spend.
When you do add them, run inference as a separate deployment with its own
node selector rather than putting a GPU on every API pod — inference is a small
fraction of total request time.

---

## Observability

Worth instrumenting, in priority order:

1. **Time split between STAC search, pixel reads, and computation.** Reads
   dominate; if they do not, something is wrong.
2. **Cache hit rate** (`/api/v1/health` reports entries). A low rate on repeated
   dashboard use means keys are varying when they should not.
3. **404 rate from analysis endpoints.** These are "no imagery", not failures —
   alert on a *change* in the rate, since a spike usually means an upstream
   collection issue rather than a bug here.
4. **Overview-level selection.** A read that falls back to full resolution is
   ~9× more bytes; log when `_overview_level` returns `None` for a fine-grained
   grid.

Structured logs go to stdout at `ATLASAI_LOG_LEVEL`.

---

## Security

- The API is read-only. It has no write endpoints, no authentication, and no
  user data — the threat model is resource exhaustion, not data exfiltration.
- **Put a rate limiter in front of it.** An unbounded caller can generate
  expensive analyses indefinitely.
- Both containers run as non-root uid 10001.
- `ANTHROPIC_API_KEY` and `ATLASAI_PC_SUBSCRIPTION_KEY` are the only secrets;
  neither is echoed by any endpoint (`/health` reports only whether they are
  set).
- The assistant sends the user's question to the Anthropic API when a key is
  configured. Its output is constrained to a routing decision validated against
  the module registry, so a prompt-injected reply cannot cause arbitrary
  analysis — but the question text does leave your infrastructure. Leave the key
  unset to keep everything local; the keyword router still works.
