ARG PY=3.10
FROM mcr.microsoft.com/devcontainers/python:1-$PY

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN --mount=type=cache,target=/var/cache \
    apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        libgdal-dev \
        gdal-bin \
        netcdf-bin \
        pandoc \
        zip unzip \
        tmux tig jq fzf silversearcher-ag \
    && true 
