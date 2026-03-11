# ── Look up default VPC networking ────────────────────────────────────────────

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "defaultForAz"
    values = ["true"]
  }

  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_security_group" "default" {
  name   = "default"
  vpc_id = data.aws_vpc.default.id
}

locals {
  subnet_ids         = length(var.subnet_ids) > 0 ? var.subnet_ids : data.aws_subnets.default.ids
  security_group_ids = length(var.security_group_ids) > 0 ? var.security_group_ids : [data.aws_security_group.default.id]
}

# ── Compute Environment ────────────────────────────────────────────────────────

resource "aws_batch_compute_environment" "odm_spot" {
  compute_environment_name = "geo-odm-compute"
  type                     = "MANAGED"
  service_role             = var.service_role_arn
  state                    = "ENABLED"

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    bid_percentage      = 100

    instance_type = [
      "c5.4xlarge",
      "c5.9xlarge",
    ]

    min_vcpus     = 0
    desired_vcpus = 0
    max_vcpus     = 256

    spot_iam_fleet_role = var.spot_fleet_role_arn
    instance_role       = var.instance_profile_arn

    subnets            = local.subnet_ids
    security_group_ids = local.security_group_ids

    tags = {
      Project     = "geo"
      Environment = var.environment
    }
  }

  tags = {
    Project     = "geo"
    Environment = var.environment
  }
}

# ── Job Queue ──────────────────────────────────────────────────────────────────

resource "aws_batch_job_queue" "odm_queue" {
  name     = "geo-odm-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.odm_spot.arn
  }

  tags = {
    Project     = "geo"
    Environment = var.environment
  }
}

# ── Job Definition: odm-run ────────────────────────────────────────────────────

resource "aws_batch_job_definition" "odm_run" {
  name = "odm-run"
  type = "container"

  container_properties = jsonencode({
    image   = var.odm_image_url
    vcpus   = 16
    memory  = 30000
    command = ["/app/run.sh"]

    jobRoleArn      = var.job_role_arn
    executionRoleArn = var.job_role_arn

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"  = "/aws/batch/odm-run"
        "awslogs-region" = var.region
      }
    }

    environment = []
  })

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = 14400
  }

  tags = {
    Project     = "geo"
    Environment = var.environment
  }
}

# ── Job Definition: rmse-calc ──────────────────────────────────────────────────

resource "aws_batch_job_definition" "rmse_calc" {
  name = "rmse-calc"
  type = "container"

  container_properties = jsonencode({
    image   = var.tools_image_url
    vcpus   = 2
    memory  = 4000
    command = ["python", "/app/rmse_calc_batch.py"]

    jobRoleArn      = var.job_role_arn
    executionRoleArn = var.job_role_arn

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"  = "/aws/batch/rmse-calc"
        "awslogs-region" = var.region
      }
    }

    environment = []
  })

  # attempts must be >= 1 per Terraform validation; spec says 0 retries meaning
  # no automatic retry — we use 1 (the minimum, meaning run once with no retry).
  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = 1800
  }

  tags = {
    Project     = "geo"
    Environment = var.environment
  }
}
