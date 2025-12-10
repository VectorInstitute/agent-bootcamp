terraform {
  required_providers {
    coder = {
      source = "coder/coder"
    }
    google = {
      source = "hashicorp/google"
    }
  }
}

provider "coder" {}

provider "google" {
  zone    = var.zone
  project = var.project
}

data "coder_provisioner" "me" {}
data "coder_workspace" "me" {}
data "coder_workspace_owner" "me" {}
data "coder_external_auth" "github" {
  id = var.github_app_id
}

locals {
  # Ensure Coder username is a valid Linux username
  username  = "coder"
  repo_name = replace(regex(".*/(.*)", var.github_repo)[0], ".git", "")
}

resource "coder_agent" "main" {
  auth = "google-instance-identity"
  arch = "amd64"
  os   = "linux"

  display_apps {
    vscode = false
  }

  startup_script = <<-EOT
    #!/bin/bash
    set -e

    # Fix permissions immediately - must be first!
    echo "Fixing permissions for /home/${local.username}"
    sudo chown -R ${local.username}:${local.username} /home/${local.username}

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/home/${local.username}/.local/bin:$PATH"

    # Clone the GitHub repository with proper error handling
    cd "/home/${local.username}"

    # Handle three scenarios:
    # 1. Directory doesn't exist - fresh clone
    # 2. Directory exists with .git - update existing repo
    # 3. Directory exists without .git - corrupted state, clean and re-clone
    if [ -d "${local.repo_name}" ]; then
      if [ -d "${local.repo_name}/.git" ]; then
        echo "Repository already exists, updating..."
        cd ${local.repo_name}
        git pull || echo "Warning: git pull failed, continuing with existing code"
      else
        echo "Directory exists but is not a git repository, cleaning up..."
        rm -rf ${local.repo_name}
        echo "Cloning repository..."
        if git clone ${var.github_repo} ${local.repo_name}; then
          echo "Repository cloned successfully"
          cd ${local.repo_name}
          git checkout ${var.github_branch}
        else
          echo "ERROR: Failed to clone repository"
          exit 1
        fi
      fi
    else
      echo "Cloning repository..."
      if git clone ${var.github_repo} ${local.repo_name}; then
        echo "Repository cloned successfully"
        cd ${local.repo_name}
        git checkout ${var.github_branch}
      else
        echo "ERROR: Failed to clone repository"
        exit 1
      fi
    fi

    # Verify we're in the correct directory with a valid repo
    if [ ! -d ".git" ]; then
      echo "ERROR: Not in a valid git repository"
      exit 1
    fi

    echo "Current directory: $(pwd)"
    echo "Directory contents: $(ls -la)"

    # Run project init steps
    echo "Creating virtual environment and installing dependencies..."
    uv venv .venv
    source .venv/bin/activate

    # Run sync synchronously and wait for completion only if pyproject.toml exists
    if [ -f "pyproject.toml" ]; then
      echo "Found pyproject.toml, installing dependencies..."
      uv sync --dev
      sync_exit_code=$?

      # Ensure sync completed successfully before proceeding
      if [ $sync_exit_code -eq 0 ]; then
        echo "Dependencies installed successfully"
      else
        echo "Warning: uv sync exited with code $sync_exit_code"
      fi
    else
      echo "No pyproject.toml found in $(pwd), skipping dependency installation"
    fi

    # Wait a moment to ensure all installations are finalized
    sleep 2

    # Run automatic onboarding
    echo "Running automatic onboarding..."
    if command -v onboard &> /dev/null; then
      onboard \
        --bootcamp-name "$BOOTCAMP_NAME" \
        --output-dir "/home/${local.username}/${local.repo_name}" \
        --test-script "/home/${local.username}/${local.repo_name}/tests/tool_tests/test_integration.py" || echo "Onboarding failed, continuing..."
    else
      echo "Onboarding CLI not found, skipping automated onboarding"
    fi

    # Configure VS Code settings
    mkdir -p "/home/${local.username}/${local.repo_name}/.vscode"
    cat > "/home/${local.username}/${local.repo_name}/.vscode/settings.json" <<'VSCODE_SETTINGS'
{
  "python.terminal.useEnvFile": true
}
VSCODE_SETTINGS

    # Configure shell to always start in repo with venv activated
    cat >> "/home/${local.username}/.bashrc" <<BASHRC

# Auto-navigate to ${local.repo_name} and activate venv
if [ -f ~/${local.repo_name}/.venv/bin/activate ]; then
    cd ~/${local.repo_name}
    source .venv/bin/activate
fi
BASHRC

    echo "Startup script ran successfully!"

  EOT

  env = {
    GIT_AUTHOR_NAME      = coalesce(data.coder_workspace_owner.me.full_name, data.coder_workspace_owner.me.name)
    GIT_AUTHOR_EMAIL     = "${data.coder_workspace_owner.me.email}"
    GIT_COMMITTER_NAME   = coalesce(data.coder_workspace_owner.me.full_name, data.coder_workspace_owner.me.name)
    GIT_COMMITTER_EMAIL  = "${data.coder_workspace_owner.me.email}"
    GITHUB_USER          = data.coder_workspace_owner.me.name
    TOKEN_SERVICE_URL    = var.token_service_url
    BOOTCAMP_NAME        = var.bootcamp_name
    FIREBASE_WEB_API_KEY = var.firebase_api_key
  }

  metadata {
    display_name = "CPU Usage"
    key          = "0_cpu_usage"
    script       = "coder stat cpu"
    interval     = 10
    timeout      = 1
  }

  metadata {
    display_name = "RAM Usage"
    key          = "1_ram_usage"
    script       = "coder stat mem"
    interval     = 10
    timeout      = 1
  }
}

module "github-upload-public-key" {
  count            = data.coder_workspace.me.start_count
  source           = "registry.coder.com/coder/github-upload-public-key/coder"
  version          = "1.0.15"
  agent_id         = coder_agent.main.id
  external_auth_id = data.coder_external_auth.github.id
}

# See https://registry.terraform.io/modules/terraform-google-modules/container-vm
module "gce-container" {
  source  = "terraform-google-modules/container-vm/google"
  version = "3.0.0"

  container = {
    image   = var.container_image
    command = ["sh"]
    args    = ["-c", "chown -R ${local.username}:${local.username} /home/${local.username} && su - ${local.username} -s /bin/bash <<'CODER_SCRIPT'\n${coder_agent.main.init_script}\nCODER_SCRIPT\n"]
    securityContext = {
      privileged : true
    }
    # Declare volumes to be mounted
    # This is similar to how Docker volumes are mounted
    volumeMounts = [
      {
        mountPath = "/cache"
        name      = "tempfs-0"
        readOnly  = false
      },
      {
        mountPath = "/home/${local.username}"
        name      = "data-disk-0"
        readOnly  = false
      },
    ]
  }
  # Declare the volumes
  volumes = [
    {
      name = "tempfs-0"

      emptyDir = {
        medium = "Memory"
      }
    },
    {
      name = "data-disk-0"

      gcePersistentDisk = {
        pdName = "data-disk-0"
        fsType = "ext4"
      }
    },
  ]
}

resource "google_compute_disk" "pd" {
  project = var.project
  name    = "coder-${data.coder_workspace.me.id}-data-disk"
  type    = "pd-ssd"
  zone    = var.zone
  size    = var.pd_size
}

resource "google_compute_instance" "dev" {
  zone         = var.zone
  count        = data.coder_workspace.me.start_count
  name         = "coder-${lower(data.coder_workspace_owner.me.name)}-${lower(data.coder_workspace.me.name)}"
  machine_type = var.machine_type
  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }
  boot_disk {
    initialize_params {
      image = module.gce-container.source_image
    }
  }
  attached_disk {
    source      = google_compute_disk.pd.self_link
    device_name = "data-disk-0"
    mode        = "READ_WRITE"
  }
  service_account {
    email  = var.service_account_email
    scopes = ["cloud-platform"]
  }
  metadata = {
    "gce-container-declaration" = module.gce-container.metadata_value
  }
  labels = {
    container-vm = module.gce-container.vm_container_label
  }
}

resource "coder_agent_instance" "dev" {
  count       = data.coder_workspace.me.start_count
  agent_id    = coder_agent.main.id
  instance_id = google_compute_instance.dev[0].instance_id
}

resource "coder_metadata" "workspace_info" {
  count       = data.coder_workspace.me.start_count
  resource_id = google_compute_instance.dev[0].id

  item {
    key   = "image"
    value = module.gce-container.container.image
  }
}

module "vscode-web" {
  count          = tobool(var.codeserver) ? data.coder_workspace.me.start_count : 0
  source         = "registry.coder.com/coder/vscode-web/coder"
  version        = "1.3.0"
  agent_id       = coder_agent.main.id
  extensions     = ["ms-python.python", "ms-python.vscode-pylance", "ms-vsliveshare.vsliveshare"]
  install_prefix = "/tmp/.vscode-web"
  folder         = "/home/coder/${local.repo_name}"
  accept_license = true
  subdomain      = false
  order          = 1
}
