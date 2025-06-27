FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update
RUN apt-get install -y \
    sudo \
    curl \
    git \
    tar \
    unzip \
    ca-certificates \
    build-essential \
    golang \
    procps \
    python3.10 \
    python3-pip \
    python3-venv

RUN python3 -m pip install jupyter uv gradio
 
ARG USER=coder
RUN useradd --groups sudo --no-create-home --shell /bin/bash ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

USER ${USER}
WORKDIR /home/${USER}