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
    type = string
    default = "false"
}

variable "codeserver" {
    type = string
    default = "true"
}

variable "streamlit" {
    type = string
    default = "false"
}

variable github_repo {
    type = string
}

variable github_branch {
    type = string
}

variable github_app_id {
    type = string
}

variable container_image {
    type = string
}

variable "service_account_email" {
    type = string
}
