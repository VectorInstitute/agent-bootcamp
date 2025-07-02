#!/bin/bash

# Redirect all the following output to /install.log
exec > /tmp/install.log 2>&1
chmod 777 /tmp/install.log

# Update and install system packages
apt-get update
apt-get install -y apt-transport-https ca-certificates curl software-properties-common members vim python3-pandas

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
    add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable"
apt-get update
apt-get install -y docker-ce

# Set some Coder-related environment variables
# TODO: This still doesn't work, fix it
export CODER_OAUTH2_GITHUB_ALLOW_SIGNUPS=true

# Install Coder
export HOME=/root
curl -L https://coder.com/install.sh | sh
systemctl enable --now coder
journalctl -u coder.service -b

# Since Coder is running as a systemd service, we need to set environment variables in the coder.service.d directory
mkdir -p /etc/systemd/system/coder.service.d
echo -e '[Service]\nEnvironment=CODER_OAUTH2_GITHUB_ALLOW_SIGNUPS=true\nEnvironment=CODER_EXTERNAL_AUTH_0_ID="primary-github"\nEnvironment=CODER_EXTERNAL_AUTH_0_TYPE=github\nEnvironment=CODER_EXTERNAL_AUTH_0_CLIENT_ID=<CLIENT_ID>\nEnvironment=CODER_EXTERNAL_AUTH_0_CLIENT_SECRET=<CLIENT_SECRET>\nEnvironment=CODER_EXTERNAL_AUTH_0_SCOPES="repo,workflow,admin:public_key"' > /etc/systemd/system/coder.service.d/override.conf

# Reload systemd to apply the changes
systemctl daemon-reload

# Restart the Coder service to apply the new environment variables
systemctl restart coder
