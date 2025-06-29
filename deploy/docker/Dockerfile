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

# Now, activate the coder user account, set as ${USER} here
USER ${USER}
WORKDIR /home/${USER}

# Setup user account but not activate it yet
RUN mkdir -p /home/${USER}/.config/gcloud && \
    chown -R ${USER}:${USER} /home/${USER}/.config

# Install the gcloud CLI, still as superuser
RUN apt-get install -y apt-transport-https ca-certificates gnupg
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && sudo apt-get install -y google-cloud-cli


# Add the agent-bootcamp repo code
# TODO: Fix this ugly hack, find a better way to grab the agent-bootcamp git repo
RUN gsutil cp gs://agent-bootcamp/agent-bootcamp-git.zip /tmp/
RUN unzip /tmp/agent-bootcamp-git.zip