# Root Terragrunt configuration.
#
# REMOTE STATE (for apply):
#   Before running `terragrunt apply`, bootstrap the state bucket and lock table:
#     AWS_PROFILE=personal terragrunt backend bootstrap --terragrunt-working-dir infra/dev/s3
#   State bucket: geo-terraform-state-<account_id>
#   Lock table:   geo-terraform-locks
#
# PLAN-ONLY TESTING (before state bucket exists):
#   The remote_state block below will attempt to create the S3 bucket automatically
#   if it does not exist. For plan-only runs without creating infrastructure,
#   run with TG_DISABLE_BACKEND=true:
#     TG_DISABLE_BACKEND=true AWS_PROFILE=personal terragrunt plan ...

locals {
  use_local_backend = tobool(get_env("TG_USE_LOCAL_BACKEND", "false"))
}

remote_state {
  backend = local.use_local_backend ? "local" : "s3"
  config = local.use_local_backend ? {
    path = "${get_terragrunt_dir()}/terraform.tfstate"
  } : {
    bucket         = "geo-terraform-state-${get_aws_account_id()}"
    key            = "${path_relative_to_include()}/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "geo-terraform-locks"
  }
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = "us-east-1"
  profile = "personal"
}
EOF
}
