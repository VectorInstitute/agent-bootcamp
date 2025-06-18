project = "coder-evaluation" # This must match your GCP project id
script_path = "startup.sh"
region = "us-central1"
zone = "us-central1-a"
machine_type = "e2-small"
service_account_email = "coder-service-account@coder-evaluation.iam.gserviceaccount.com"
public_key_path = "~/.ssh/id_rsa.pub"
vm_names = ["coder-entrypoint"]