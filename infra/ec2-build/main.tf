# One-shot EC2 builder — clones an ODM fork, builds the Docker image, pushes to ECR.
#
# ── USAGE ──────────────────────────────────────────────────────────────────────
#
# STEP 1 — Apply (launches builder, build starts automatically):
#   terraform apply \
#     [-var="notify_email=you@example.com"]
#
# STEP 2 — Save SSH key (first time only, after apply):
#   terraform output -raw private_key_pem > ~/.ssh/geo-odm-build.pem && chmod 600 ~/.ssh/geo-odm-build.pem
#
# STEP 3 — Watch build log (optional):
#   ssh -i ~/.ssh/geo-odm-build.pem ec2-user@$(terraform output -raw public_ip)
#   tail -f /var/log/odm-build.log
#
# STEP 4 — Instance self-terminates ~2 min after a successful push.
#          On failure it stays up for debugging; destroy manually when done.
#
# STEP 5 — Tear down:
#   terraform destroy
# ───────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
    tls = { source = "hashicorp/tls", version = "~> 4.0" }
  }
  backend "local" {}
}

provider "aws" {
  region = var.region
}

# ── Variables ──────────────────────────────────────────────────────────────────

variable "region" {
  description = "AWS region"
  default     = "us-west-2"
}

variable "instance_type" {
  description = "EC2 instance type for the build. c5.4xlarge (16 vCPU / 32 GB) is the default — enough for a full ODM parallel build."
  default     = "c5.4xlarge"
}

variable "odm_repo" {
  description = "Git URL of the ODM fork to build."
  default     = "https://github.com/jrstear/ODM"
}

variable "odm_branch" {
  description = "Branch (or tag/SHA) to build."
  default     = "checkpoint_rmse"
}

variable "ecr_repo" {
  description = "Full ECR repository URL (without tag). Must already exist."
  default     = "658302145097.dkr.ecr.us-west-2.amazonaws.com/odm"
}

variable "image_tag" {
  description = "Docker image tag to push. Defaults to odm_branch."
  default     = ""  # empty = use odm_branch
}

variable "notify_email" {
  description = "Email address for build completion / failure alerts. Empty = no subscription."
  default     = ""
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH in."
  default     = "0.0.0.0/0"
}

# ── Locals ─────────────────────────────────────────────────────────────────────

locals {
  tag = var.image_tag != "" ? var.image_tag : var.odm_branch
}

# ── SSH Key ────────────────────────────────────────────────────────────────────

resource "tls_private_key" "build" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "build" {
  key_name   = "geo-odm-build"
  public_key = tls_private_key.build.public_key_openssh
  tags       = { Project = "geo", Purpose = "odm-build" }
}

# ── IAM: ECR push + pull ───────────────────────────────────────────────────────

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "build" {
  name               = "geo-odm-build-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = { Project = "geo", Purpose = "odm-build" }
}

data "aws_iam_policy_document" "ecr_push" {
  statement {
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
    ]
    resources = ["arn:aws:ecr:${var.region}:*:repository/*"]
  }
}

resource "aws_iam_role_policy" "build_ecr" {
  name   = "geo-odm-build-ecr"
  role   = aws_iam_role.build.name
  policy = data.aws_iam_policy_document.ecr_push.json
}

data "aws_iam_policy_document" "sns_publish" {
  statement {
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.build_alerts.arn]
  }
}

resource "aws_iam_role_policy" "build_sns" {
  name   = "geo-odm-build-sns"
  role   = aws_iam_role.build.name
  policy = data.aws_iam_policy_document.sns_publish.json
}

resource "aws_iam_instance_profile" "build" {
  name = "geo-odm-build-profile"
  role = aws_iam_role.build.name
}

# ── Security group ─────────────────────────────────────────────────────────────

resource "aws_security_group" "build" {
  name        = "geo-odm-build-sg"
  description = "SSH ingress + all egress for ODM Docker build"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Project = "geo", Purpose = "odm-build" }
}

# ── AMI — Amazon Linux 2023 (x86_64) ──────────────────────────────────────────

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
  filter {
    name   = "state"
    values = ["available"]
  }
}

# ── SNS alerts ─────────────────────────────────────────────────────────────────

resource "aws_sns_topic" "build_alerts" {
  name = "geo-odm-build-alerts"
  tags = { Project = "geo", Purpose = "odm-build" }
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.notify_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.build_alerts.arn
  protocol  = "email"
  endpoint  = var.notify_email
}

# ── EC2 instance ───────────────────────────────────────────────────────────────

resource "aws_instance" "build" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.build.name
  key_name               = aws_key_pair.build.key_name
  vpc_security_group_ids = [aws_security_group.build.id]

  root_block_device {
    volume_size           = 80    # GB — ODM build artifacts + Docker layers
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = base64encode(<<-USERDATA
    #!/bin/bash
    set -euo pipefail
    exec > /var/log/odm-build.log 2>&1

    ECR_REPO="${var.ecr_repo}"
    ODM_REPO="${var.odm_repo}"
    ODM_BRANCH="${var.odm_branch}"
    IMAGE_TAG="${local.tag}"
    SNS_TOPIC="${aws_sns_topic.build_alerts.arn}"
    REGION="${var.region}"
    FULL_IMAGE="$${ECR_REPO}:$${IMAGE_TAG}"

    notify() {
      aws sns publish \
        --topic-arn "$${SNS_TOPIC}" \
        --subject "$1" \
        --message "$2" \
        --region "$${REGION}" || true
    }

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Installing dependencies"
    dnf update -y
    dnf install -y docker git
    systemctl enable --now docker

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Logging into ECR"
    aws ecr get-login-password --region "$${REGION}" \
      | docker login --username AWS --password-stdin "$${ECR_REPO}"

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Cloning $${ODM_REPO} @ $${ODM_BRANCH}"
    git clone --depth 1 --branch "$${ODM_BRANCH}" "$${ODM_REPO}" /tmp/ODM

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Building $${FULL_IMAGE}"
    notify "ODM build started: $${IMAGE_TAG}" \
      "Building $${FULL_IMAGE} on $(hostname). SSH in and tail -f /var/log/odm-build.log to watch."

    if docker build --platform linux/amd64 -t "$${FULL_IMAGE}" /tmp/ODM; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Build complete — pushing to ECR"
      docker push "$${FULL_IMAGE}"
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Push complete"
      notify "ODM build complete: $${IMAGE_TAG}" \
        "Image pushed to $${FULL_IMAGE}. Instance shutting down in 2 minutes."
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Done. Shutting down in 2 minutes."
      /sbin/shutdown -h +2
    else
      EXIT=$?
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Build FAILED (exit $${EXIT}) — instance remains up for debugging"
      notify "ODM build FAILED: $${IMAGE_TAG}" \
        "docker build exited $${EXIT}. SSH in to investigate: tail -f /var/log/odm-build.log"
    fi
  USERDATA
  )

  tags = {
    Project  = "geo"
    Purpose  = "odm-build"
    Name     = "geo-odm-build"
    ODMBranch = var.odm_branch
  }
}

# ── Outputs ────────────────────────────────────────────────────────────────────

output "public_ip" {
  value       = aws_instance.build.public_ip
  description = "SSH: ssh -i ~/.ssh/geo-odm-build.pem ec2-user@<IP>"
}

output "instance_id" {
  value = aws_instance.build.id
}

output "image_uri" {
  value       = "${var.ecr_repo}:${local.tag}"
  description = "ECR image URI — use as odm_image in infra/ec2/ terraform apply"
}

output "private_key_pem" {
  value       = tls_private_key.build.private_key_pem
  sensitive   = true
  description = "Save with: terraform output -raw private_key_pem > ~/.ssh/geo-odm-build.pem && chmod 600 ~/.ssh/geo-odm-build.pem"
}

output "instructions" {
  value = <<-EOT

    ── BUILD IN PROGRESS ─────────────────────────────────────────────────────────

    Image:   ${var.ecr_repo}:${local.tag}
    Branch:  ${var.odm_branch}

    ── SSH / WATCH LOG ───────────────────────────────────────────────────────────

    terraform output -raw private_key_pem > ~/.ssh/geo-odm-build.pem && chmod 600 ~/.ssh/geo-odm-build.pem
    ssh -i ~/.ssh/geo-odm-build.pem ec2-user@${aws_instance.build.public_ip}
    tail -f /var/log/odm-build.log

    ── AFTER BUILD COMPLETES ─────────────────────────────────────────────────────

    # Use this image in infra/ec2/:
    cd ../ec2
    terraform apply \
      -var="project=bsn/aztec6" \
      -var="odm_image=${var.ecr_repo}:${local.tag}" \
      -var="notify_email=you@example.com"

    ── TEAR DOWN ─────────────────────────────────────────────────────────────────

    terraform destroy   # instance self-terminates on success, but still cleans up SG/IAM/etc.

  EOT
}
