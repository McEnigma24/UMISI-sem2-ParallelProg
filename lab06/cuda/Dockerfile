# ==========================================
# 0. BAZA
# ==========================================
# CUDA w obrazie nie może być nowsza niż sterownik z nvidia-smi (inaczej binarka się nie odpali).
# przypięcie konkretnej wersji (reprodukowalny build):
#   docker build --build-arg CUDA_TAG=13.3.1 --build-arg UBUNTU_TAG=26.04 ...
ARG CUDA_TAG=13.3.1
ARG UBUNTU_TAG=26.04

# ==========================================
# 1. WARSTWA: RUNTIME BASE (common ground)
# ==========================================
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu${UBUNTU_TAG} AS runtime-base
ENV DEBIAN_FRONTEND=noninteractive \
    NEEDRESTART_MODE=a \
    TERM=xterm-256color

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libcurl4 \
        libgomp1 \
        zlib1g && \
    rm -rf /var/lib/apt/lists/*

# ==========================================
# 2. WARSTWA: DEV
# ==========================================
# Nie FROM runtime-base — nvcc i nagłówki CUDA są tylko w obrazie -devel.
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu${UBUNTU_TAG} AS dev-env

ENV DEBIAN_FRONTEND=noninteractive \
    NEEDRESTART_MODE=a \
    TERM=xterm-256color

# python3-matplotlib: wykresy z JSON-a Google Benchmark (apt, nie pip — PEP 668)
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        tar \
        make \
        cmake \
        ninja-build \
        build-essential \
        gcc \
        g++ \
        clang \
        libomp-dev \
        libbenchmark-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libprotobuf-dev \
        protobuf-compiler \
        git \
        ca-certificates \
        python3 \
        python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*

# kontener chodzi jako użytkownik hosta i nie ma swojego $HOME w obrazie,
# więc matplotlib musi dostać jawnie zapisywalny katalog na cache
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

# ==========================================
# 3. ETAP: RUNNER (wariant izolowany)
# ==========================================
# Wszystko wkopiowane do obrazu, więc odpala się gdziekolwiek -> docker/run_isolated.sh
# Wariant "na repo" (mount jak w dev-env) nie ma tu swojego etapu,
# bo używa gołego runtime-base -> docker/run.sh
FROM runtime-base AS runner
WORKDIR /app

RUN mkdir -p exe log output input run_time_config build
COPY ./input /app/input
COPY ./run_time_config /app/run_time_config
COPY ./build/*.exe /app/build/
COPY ./build/*.a /app/build/
COPY ./build/*.so /app/build/
COPY ./build/*.dylib /app/build/
ENV LD_LIBRARY_PATH=/app/build
CMD ["/bin/sh", "-c", "exec /app/build/*.exe"]