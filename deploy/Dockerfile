# AeroNet, one container: the FastAPI backend serves the built React page and the API on one port.
#
# Default build: the exported data is included WITH NNfoil's finite-volume (FVM) reference data (reference curves, same-mesh curves and the
# bundled sample), which the NNfoil author allowed us to show (docs/DEPLOY.md, "Licence"); the pages carry the credit.
# Strict build (--build-arg WITHHOLD_FVM=1): the image is built from a STRIPPED copy of the data (deploy/make_data.py, verified in the build)
# and the server runs with AERONET_WITHHOLD_FVM_DATA=1, so it holds and serves no FVM Cp data at all.
#
#   docker build -f deploy/Dockerfile -t aeronet .                              # default
#   docker build -f deploy/Dockerfile --build-arg WITHHOLD_FVM=1 -t aeronet-strict .   # strict (needs a bundle made with prepare-public -Withhold)
#   docker run --rm -p 8000:8000 aeronet                                         # then open http://localhost:8000
#
# Other build argument: --build-arg WITH_NEURALFOIL=0 leaves out NeuralFoil (smaller image; the viscous card then says it is not installed).

ARG WITHHOLD_FVM=0

# ---- 1. the page --------------------------------------------------------------------------------
FROM node:22-slim AS web
ARG WITHHOLD_FVM
WORKDIR /web
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci
COPY app/frontend/ ./
# the saved results must match the mode (see scripts/check-public-bundle.mjs); a mismatch fails the build
RUN if [ "$WITHHOLD_FVM" = "1" ]; then node scripts/check-public-bundle.mjs --withhold; else node scripts/check-public-bundle.mjs; fi && npm run build

# ---- 2. the data: complete, or (strict) a stripped copy that is verified to hold no FVM curve ------
FROM python:3.13-slim AS data
ARG WITHHOLD_FVM
WORKDIR /w
RUN pip install --no-cache-dir "numpy>=2.1"
COPY app/backend/aeronet ./app/backend/aeronet
COPY app/backend/data ./app/backend/data
COPY deploy/make_data.py ./deploy/make_data.py
RUN if [ "$WITHHOLD_FVM" = "1" ]; then \
      python deploy/make_data.py --src app/backend/data --out /out && python deploy/make_data.py --check /out; \
    else cp -r app/backend/data /out; fi

# ---- 3. the server -------------------------------------------------------------------------------
FROM python:3.13-slim AS app
ARG WITH_NEURALFOIL=1
ARG WITHHOLD_FVM
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    AERONET_WITHHOLD_FVM_DATA=${WITHHOLD_FVM} \
    AERONET_MAX_HEAVY=2 \
    AERONET_HEAVY_PER_MIN=30
WORKDIR /app/backend
COPY app/backend/requirements-deploy.txt app/backend/requirements-viscous.txt ./
RUN pip install -r requirements-deploy.txt \
 && if [ "$WITH_NEURALFOIL" = "1" ]; then pip install -r requirements-viscous.txt; fi
COPY app/backend/aeronet ./aeronet
COPY --from=data /out ./data
COPY --from=web /web/dist /app/frontend/dist
RUN useradd --create-home --uid 10001 aeronet
USER aeronet
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8000'),timeout=4)"
# --proxy-headers: behind a platform's proxy the real client address is used for the per-address limit
CMD ["sh", "-c", "exec uvicorn aeronet.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
