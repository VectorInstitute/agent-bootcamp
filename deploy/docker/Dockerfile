FROM ubuntu:24.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    sudo \
    curl \
    git \
    jq \
    tar \
    unzip \
    ca-certificates \
    build-essential \
    golang \
    procps \
    python3 \
    python3-pip \
    python3-venv \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*


# Setup the coder user account
ARG USER=coder
RUN useradd --groups sudo --shell /bin/bash ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

# Setup user account and home directory in the image
RUN mkdir -p /home/${USER}/.config/gcloud && \
    chown -R ${USER}:${USER} /home/${USER}

# Note: We don't set USER here because the container needs to run as root
# to fix permissions on the mounted volume. The container command will
# chown /home/coder and then su to the coder user.