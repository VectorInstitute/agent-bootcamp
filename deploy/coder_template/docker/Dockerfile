FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
    sudo \
    curl \
    git \
    tar \
    unzip \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG USER=coder
RUN useradd --groups sudo --no-create-home --shell /bin/bash ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

USER ${USER}
WORKDIR /home/${USER}