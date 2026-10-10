# ==========================================
# 0. BAZA
# ==========================================
# tag 'latest' na Docker Hub zawsze wskazuje najnowsze Ubuntu LTS
# przypięcie konkretnej wersji (reprodukowalny build):
#   docker build --build-arg UBUNTU_TAG=26.04 ...
ARG UBUNTU_TAG=latest

# ==========================================
# 1. WARSTWA: RUNTIME BASE (common ground)
# ==========================================
FROM ubuntu:${UBUNTU_TAG} AS runtime-base
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
FROM runtime-base AS dev-env

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
