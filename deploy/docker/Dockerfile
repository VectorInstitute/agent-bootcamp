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

# Install the gcloud CLI
RUN apt-get install -y apt-transport-https ca-certificates gnupg
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && sudo apt-get install -y google-cloud-cli

ARG USER=coder
RUN useradd --groups sudo --no-create-home --shell /bin/bash ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

USER ${USER}
WORKDIR /home/${USER}