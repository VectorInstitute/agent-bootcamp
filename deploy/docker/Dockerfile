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
    && rm -rf /var/lib/apt/lists/*


# Setup the coder user account
ARG USER=coder
RUN useradd --groups sudo --shell /bin/bash ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

# Setup user account but not activate it yet
RUN mkdir /home/${USER} && \
    chown -R ${USER}:${USER} /home/${USER}
RUN mkdir -p /home/${USER}/.config/gcloud && \
    chown -R ${USER}:${USER} /home/${USER}/.config

# Install VS code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh -s -- --method=standalone --prefix=/tmp/code-server

# We want the following commands to run as a regular user, so activate the coder account
USER ${USER}
WORKDIR /home/${USER}