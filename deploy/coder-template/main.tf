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

locals {
    # Ensure Coder username is a valid Linux username
    username = "coder"
    gcp_project_id = "coder-evaluation"
    gcp_zone = "us-central1-a"
    github_repo = "https://github.com/VectorInstitute/agent-bootcamp"
    github_branch = "master"
    repo_name = "agent-bootcamp"
    container_image = "us-central1-docker.pkg.dev/coder-evaluation/agent-bootcamp/agent-workspace:latest"
}

provider "coder" {}

provider "google" {
    zone    = local.gcp_zone
    project = local.gcp_project_id
}

data "coder_provisioner" "me" {}
data "coder_workspace" "me" {}
data "coder_workspace_owner" "me" {}

resource "coder_agent" "main" {
    auth           = "google-instance-identity"
    arch           = "amd64"
    os             = "linux"
    startup_script = <<-EOT
        #!/bin/bash
        set -e

        export PATH="/home/${local.username}/.local/bin:$PATH"

        echo "Changing permissions of /home/${local.username} folder"
        sudo chown -R ${local.username}:${local.username} /home/${local.username}

        echo "Installing Code Server"

        # Install and start code-server
        sudo curl -fsSL https://code-server.dev/install.sh | sh -s -- --method=standalone --prefix=/tmp/code-server
        /tmp/code-server/bin/code-server --auth none --port 13337 >/tmp/code-server.log 2>&1 &

        echo "Running project init script"

        # TODO: Fix this ugly hack, find a better way to grab the agent-bootcamp git repo
        gsutil cp gs://agent-bootcamp/agent-bootcamp-git.zip /tmp/
        unzip /tmp/agent-bootcamp-git.zip

        cd /home/${local.username}/agent-bootcamp/
        sudo bash deploy/coder-template/init.sh

        echo "Startup script ran successfully!"

    EOT

    env = {
        GIT_AUTHOR_NAME     = coalesce(data.coder_workspace_owner.me.full_name, data.coder_workspace_owner.me.name)
        GIT_AUTHOR_EMAIL    = "${data.coder_workspace_owner.me.email}"
        GIT_COMMITTER_NAME  = coalesce(data.coder_workspace_owner.me.full_name, data.coder_workspace_owner.me.name)
        GIT_COMMITTER_EMAIL = "${data.coder_workspace_owner.me.email}"
    }
}

# See https://registry.terraform.io/modules/terraform-google-modules/container-vm
module "gce-container" {
    source  = "terraform-google-modules/container-vm/google"
    version = "3.0.0"

    container = {
        image   = local.container_image
        command = ["sh"]
        args    = ["-c", coder_agent.main.init_script]
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

resource "google_storage_bucket_iam_member" "allow_sa_read_bucket" {
    bucket = "agent-bootcamp"
    role   = "roles/storage.objectViewer"
    member = "serviceAccount:coder-service-account@coder-evaluation.iam.gserviceaccount.com"
}


resource "google_compute_disk" "pd" {
    project = local.gcp_project_id
    name    = "coder-${data.coder_workspace.me.id}-data-disk"
    type    = "pd-ssd"
    zone    = local.gcp_zone
    size    = 10
}

resource "google_compute_instance" "dev" {
    zone         = local.gcp_zone
    count        = data.coder_workspace.me.start_count
    name         = "coder-${lower(data.coder_workspace_owner.me.name)}-${lower(data.coder_workspace.me.name)}"
    machine_type = "e2-small"
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
        email  = "coder-service-account@coder-evaluation.iam.gserviceaccount.com"
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

resource "coder_app" "jupyter" {
    agent_id     = coder_agent.main.id
    slug         = "jupyter"
    display_name = "JupyterLab"
    url          = "http://localhost:8888"
    icon         = "/icon/jupyter.svg"
    share        = "owner"
    subdomain    = true

    healthcheck {
        url       = "http://localhost:8888/api"
        interval  = 5
        threshold = 10
    }
}

resource "coder_app" "code-server" {
    agent_id     = coder_agent.main.id
    slug         = "code-server"
    display_name = "code-server"
    url          = "http://localhost:13337/?folder=/home/${local.username}/${local.repo_name}"
    icon         = "/icon/code.svg"
    subdomain    = false
    share        = "owner"

    healthcheck {
        url       = "http://localhost:13337/healthz"
        interval  = 5
        threshold = 6
    }
}

resource "coder_app" "gradio-app" {
    agent_id     = coder_agent.main.id
    slug         = "gradio-app"
    display_name = "Gradio"
    url          = "http://localhost:8501"
    icon         = "https://icon.icepanel.io/Technology/svg/Gradio.svg"
    subdomain    = false
    share        = "owner"

    healthcheck {
        url       = "http://localhost:8501/healthz"
        interval  = 5
        threshold = 6
    }
}