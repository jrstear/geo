terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Local state — this is a single long-lived bucket, no need for remote state.
  # If you later want S3-backed state: terraform state mv + backend reconfigure.
  backend "local" {}
}

provider "aws" {
  region  = var.region
  profile = "personal"
}

# ── Variables ──────────────────────────────────────────────────────────────────

variable "bucket_name" {
  description = "S3 bucket name. Must be globally unique — change if 'stratus-jrstear' is taken."
  type        = string
  default     = "stratus-jrstear"
}

variable "region" {
  description = "AWS region. Oregon (us-west-2) chosen for West-Coast upload speed and future EC2 proximity."
  type        = string
  default     = "us-west-2"
}

# ── Bucket ─────────────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "stratus" {
  bucket = var.bucket_name

  tags = {
    Project = "geo"
    Purpose = "flight-data-storage"
  }
}

resource "aws_s3_bucket_versioning" "stratus" {
  bucket = aws_s3_bucket.stratus.id
  versioning_configuration {
    status = "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "stratus" {
  bucket = aws_s3_bucket.stratus.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "stratus" {
  bucket = aws_s3_bucket.stratus.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── Outputs ────────────────────────────────────────────────────────────────────

output "bucket_name" {
  value = aws_s3_bucket.stratus.id
}

output "bucket_arn" {
  value = aws_s3_bucket.stratus.arn
}

output "region" {
  value = var.region
}

output "upload_command" {
  description = "Example aws s3 sync commands for uploading project data."
  value = <<-EOT
    # Red Rocks nadir images (~6,400 JPGs):
    aws s3 sync "/Volumes/Stratus JRS/BSN/Red Rocks/raw/nadir" \
      s3://${aws_s3_bucket.stratus.id}/bsn/red-rocks/raw/nadir/ \
      --exclude "*.MRK" --profile personal

    # Red Rocks GCP file:
    aws s3 cp ~/stratus/redrocks/gcp_confirmed.txt \
      s3://${aws_s3_bucket.stratus.id}/bsn/red-rocks/gcp_confirmed.txt \
      --profile personal

    # Aztec nadir images (tag when ready):
    aws s3 sync "/path/to/aztec/raw" \
      s3://${aws_s3_bucket.stratus.id}/bsn/aztec/raw/ \
      --profile personal
  EOT
}
