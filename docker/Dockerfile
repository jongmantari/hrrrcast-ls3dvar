# hrrrcast-ls3dvar: HRRRCast-live fork + LS-3DVar (blocks control, no PCA) + realtime obsdb
FROM mambaorg/micromamba:1.5.10-jammy

USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends \
      git ca-certificates curl tzdata && rm -rf /var/lib/apt/lists/*

COPY --chown=$MAMBA_USER:$MAMBA_USER aws/conda-linux-64.lock /tmp/env.lock
RUN micromamba install -y -n base -f /tmp/env.lock && micromamba clean -a -y && rm /tmp/env.lock
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -c "import requests, boto3, yaml, netCDF4, pandas, xarray, numcodecs" 2>/dev/null || pip install --no-cache-dir requests boto3 pyyaml netCDF4 pandas xarray numcodecs

COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/hrrrcast
RUN chmod +x /opt/hrrrcast/docker/entrypoint.sh /opt/hrrrcast/run_cycle.sh /opt/hrrrcast/scripts/realtime/*.sh \
 && chmod -R a+rwX /opt/hrrrcast   # container may run as the host uid (compose user:)

ENV HRRRCAST_REPO=/opt/hrrrcast \
    PYTHONPATH=/opt/hrrrcast/wxldas:/opt/hrrrcast \
    DATA=/data \
    DOMAIN=socal \
    CUDA_CACHE_PATH=/cache/nv \
    CUDA_CACHE_MAXSIZE=4294967296 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    PYTHONUNBUFFERED=1 \
    MODELS=/models \
    HOME=/tmp \
    HRRRCAST_NO_CONDA=1
WORKDIR /opt/hrrrcast
VOLUME ["/models", "/data", "/cache"]
USER $MAMBA_USER
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/opt/hrrrcast/docker/entrypoint.sh"]
CMD ["check"]
