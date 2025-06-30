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

# Install python libraries
RUN python3 -m pip install jupyter uv gradio

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

# Install the gcloud CLI
RUN apt-get install -y apt-transport-https ca-certificates gnupg
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && sudo apt-get install -y google-cloud-cli

# Install VS code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh -s -- --method=standalone --prefix=/tmp/code-server

# Add the agent-bootcamp repo code
# TODO: Fix this ugly hack, find a better way to grab the agent-bootcamp git repo
RUN apt-get update && apt-get install -y ca-certificates curl && \
    curl -v https://storage.googleapis.com/agent-bootcamp/agent-bootcamp-git.zip

RUN curl -o /tmp/agent-bootcamp-git.zip https://storage.googleapis.com/agent-bootcamp/agent-bootcamp-git.zip

# We want the following commands to run as a regular user, so activate the coder account
USER ${USER}
WORKDIR /home/${USER}
RUN unzip /tmp/agent-bootcamp-git.zip