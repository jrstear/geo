resource "aws_s3_bucket" "odm" {
  bucket = "geo-odm-${var.account_id}"

  tags = {
    Project     = "geo"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "odm" {
  bucket = aws_s3_bucket.odm.id

  versioning_configuration {
    status = "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "odm" {
  bucket = aws_s3_bucket.odm.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "odm" {
  bucket = aws_s3_bucket.odm.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Expire run output objects after 90 days.
# S3 lifecycle prefix matching is literal (no wildcards). Objects written by run.sh
# are placed under projects/<project_id>/runs/<run_id>/outputs/; they are tagged
# with Expiry=run-output at upload time so this tag-based rule matches them.
resource "aws_s3_bucket_lifecycle_configuration" "odm" {
  bucket = aws_s3_bucket.odm.id

  rule {
    id     = "expire-run-outputs"
    status = "Enabled"

    filter {
      tag {
        key   = "Expiry"
        value = "run-output"
      }
    }

    expiration {
      days = 90
    }
  }
}
