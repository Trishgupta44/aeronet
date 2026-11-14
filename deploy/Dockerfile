# AeroNet, one container: the FastAPI backend serves the built React page and the API on one port.
# Public build: no NNfoil FVM data inside (AERONET_WITHHOLD_FVM_DATA=1); the results folder is the slim one from deploy/make_data.py.
#
#   .\deploy\prepare-public.ps1 -ResultsDir <results folder>      # once, and whenever results change
#   docker build -f deploy/Dockerfile -t aeronet .
#   docker run --rm -p 8000:8000 aeronet                          # then open http://localhost:8000
#
# Build argument: --build-arg WITH_NEURALFOIL=0 leaves out NeuralFoil (smaller image; the viscous card then says it is not installed).

# ---- 1. the page --------------------------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci
COPY app/frontend/ ./
# fails when the saved results still contain FVM data; then run deploy/prepare-public.ps1
RUN node scripts/check-public-bundle.mjs && npm run build

# ---- 2. the server -------------------------------------------------------------------------------
FROM python:3.13-slim AS app
ARG WITH_NEURALFOIL=1
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    AERONET_WITHHOLD_FVM_DATA=1 \
    AERONET_RESULTS_DIR=/app/data/results \
    AERONET_MAX_HEAVY=2 \
    AERONET_HEAVY_PER_MIN=30
WORKDIR /app/backend
COPY app/backend/requirements-deploy.txt app/backend/requirements-viscous.txt ./
RUN pip install -r requirements-deploy.txt \
 && if [ "$WITH_NEURALFOIL" = "1" ]; then pip install -r requirements-viscous.txt; fi
COPY app/backend/aeronet ./aeronet
COPY --from=web /web/dist /app/frontend/dist
COPY deploy/data/results /app/data/results
RUN useradd --create-home --uid 10001 aeronet
USER aeronet
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8000'),timeout=4)"
# --proxy-headers: behind a platform's proxy the real client address is used for the per-address limit
CMD ["sh", "-c", "exec uvicorn aeronet.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
