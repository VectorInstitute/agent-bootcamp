# Agentic AI Bootcamp Deployment

The Agentic AI Bootcamp is leveraging the Coder platform (https://coder.com) to provide a cloud development environment for bootcamp participants. These deployment scripts use Terraform for a simple, turnkey deployment of a Coder entrypoint hosted on Google Cloud Platform (GCP).

Once Coder is deployed, participants will be able to log in using their Github account and create personal workspaces. Each workspace will be hosted in a new GCP virtual machine. The deployment of these workspace VMs will be managed by Coder, and the deployment scripts for these workspaces is provided in the [/coder_template](https://github.com/VectorInstitute/agent-bootcamp/tree/deploy/deploy/coder_template) subfolder.

## Setup

First, install all the following software on your local environment:
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- [Terraform](https://developer.hashicorp.com/terraform/install)
- [Python 3.10+](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/engine/install/)

### gcloud environment setup

    gcloud init
    gcloud auth login
    gcloud auth application-default login
    gcloud config set project <gcp-project-name>

### GCP Service Account setup

Coder requires a service account in GCP that will be connected to this VM. There are some clear and simple instructions here:

https://github.com/coder/coder/tree/main/examples/templates/gcp-linux#readme

## Terraform Deployment

### Start the instance

    terraform init -var-file=terraform.tfvars
    terraform apply -var-file=terraform.tfvars

### Stop the instance

    terraform destroy -var-file=../terraform.tfvars