variable "project" {
  type = string
}

variable "region" {
  type = string
}

variable "zone" {
  type = string
}

variable "machine_type" {
  type = string
}

variable "pd_size" {
  type = number
}

variable "jupyterlab" {
  type    = string
  default = "false"
}

variable "codeserver" {
  type    = string
  default = "true"
}

variable "streamlit" {
  type    = string
  default = "false"
}

variable "github_repo" {
  type = string
}

variable "github_branch" {
  type = string
}

variable "github_app_id" {
  type = string
}

variable "container_image" {
  type = string
}

variable "service_account_email" {
  type = string
}

variable "token_service_url" {
  type        = string
  description = "URL of the Firebase token generation service"
}

variable "bootcamp_name" {
  type        = string
  description = "Name of the bootcamp (e.g., agent-bootcamp)"
}

variable "firebase_api_key" {
  type        = string
  description = "Firebase Web API key for token exchange"
}
