# Rasterio ships manylinux wheels with GDAL bundled, so no system GDAL install
# is needed — which keeps this image at a few hundred MB instead of ~2 GB.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# curl is for the healthcheck; nothing else is needed at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# GDAL tuning for reading remote COGs. These matter more than they look: without
# the readdir setting, GDAL issues an HTTP request per sibling object in the
# bucket prefix before reading a single pixel.
ENV GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.TIF,.tiff,.jp2 \
    GDAL_HTTP_MULTIRANGE=YES \
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES \
    VSI_CACHE=TRUE \
    VSI_CACHE_SIZE=33554432

# Run unprivileged — the service only ever reads remote data and writes a cache.
RUN useradd --create-home --uid 10001 atlas \
    && mkdir -p /app/.cache \
    && chown -R atlas:atlas /app
USER atlas

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
