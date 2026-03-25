terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Local state — ECR repos are long-lived, no need for remote state.
  backend "local" {}
}

provider "aws" {
  region = var.region
}

# ── Variables ──────────────────────────────────────────────────────────────────

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-west-2"
}

# ── ECR Repositories ───────────────────────────────────────────────────────────

resource "aws_ecr_repository" "odm" {
  name                 = "odm"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = {
    Project = "geo"
    Purpose = "custom-odm-images"
  }
}

# Keep only the 5 most recent images per tag prefix to limit storage costs.
resource "aws_ecr_lifecycle_policy" "odm" {
  repository = aws_ecr_repository.odm.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# ── Outputs ────────────────────────────────────────────────────────────────────

output "repository_url" {
  value       = aws_ecr_repository.odm.repository_url
  description = "Full ECR URL — use as IMAGE in docker tag/push and Terraform EC2 var."
}

output "registry_id" {
  value = aws_ecr_repository.odm.registry_id
}

output "login_command" {
  description = "Run this to authenticate Docker to ECR."
  value       = "aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin ${aws_ecr_repository.odm.repository_url}"
}
