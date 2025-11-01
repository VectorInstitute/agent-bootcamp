# Agentic AI Bootcamp

This template provides the software environment needed to run Vector Institute's Agentic AI Bootcamp.

## Prerequisites

#### Authentication

This template assumes that coderd is run in an environment that is authenticated
with Google Cloud. For example, run `gcloud auth application-default login` to
import credentials on the system and user running coderd. For other ways to
authenticate [consult the Terraform
docs](https://registry.terraform.io/providers/hashicorp/google/latest/docs/guides/getting_started#adding-credentials).

Coder requires a Google Cloud Service Account to provision workspaces. To create
a service account:

1. Navigate to the [CGP
   console](https://console.cloud.google.com/projectselector/iam-admin/serviceaccounts/create),
   and select your Cloud project (if you have more than one project associated
   with your account)

1. Provide a service account name (this name is used to generate the service
   account ID)

1. Click **Create and continue**, and choose the following IAM roles to grant to
   the service account:

   - Compute Admin
   - Service Account User

   Click **Continue**.

1. Click on the created key, and navigate to the **Keys** tab.

1. Click **Add key** > **Create new key**.

#### Development environment Docker image

Build and push the Docker image to Google Cloud Artifact Registry using this [documentation](https://cloud.google.com/build/docs/build-push-docker-image). Ensure that the repository is created in the same GCP Cloud project as that where the service account was created. This is for Coder service account to be able to access the image while building the Coder workspace.

#### GitHub External Authentication

1. Create a GitHub App and complete the coderd environment setup by following the steps mentioned [here](https://coder.com/docs/admin/external-auth#github).

1. Use the value set for `CODER_EXTERNAL_AUTH_0_ID` environment variable as the value for the template variable `github_app_id`
