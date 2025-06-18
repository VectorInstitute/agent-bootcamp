provider "google" {
    project = var.project
    region  = var.region
    zone    = var.zone
}

# Set the GCP project apis first. This needs to happen before creating a new instance.
resource "google_project_service" "cloud_resource_manager_api" {
    project            = var.project
    service            = "cloudresourcemanager.googleapis.com"
    disable_on_destroy = false
}

resource "google_project_service" "compute" {
    project            = var.project
    service            = "compute.googleapis.com"
    disable_on_destroy = false
}

resource "google_project_service" "compute_api" {
    project            = var.project
    service            = "compute.googleapis.com"
    disable_on_destroy = false
}

# Create the compute instances
resource "google_compute_instance" "server" {

    # Set up multiple instances with different variable values
    for_each    = toset(var.vm_names)
    name         = each.value

    # Many of our instances use the same values (ie. project, zone)
    project      = var.project
    machine_type = var.machine_type
    zone         = var.zone
    tags         = ["${var.project}-server", "webserver-fw", "allow-ssh-iap"]
    allow_stopping_for_update = true

    boot_disk {
        initialize_params {
            image = "ubuntu-2204-lts"
        }
    }

    network_interface {
        network = "default"
        access_config {}  # Creates an external IP
    }

    service_account {
        email  = var.service_account_email
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    }

    metadata_startup_script = file(var.script_path)

    depends_on = [ google_project_service.cloud_resource_manager_api, google_project_service.compute_api ]
}


