# ── Batch Service Role ─────────────────────────────────────────────────────────

data "aws_iam_policy_document" "batch_service_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["batch.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_service" {
  name               = "geo-batch-service-role"
  assume_role_policy = data.aws_iam_policy_document.batch_service_trust.json

  tags = {
    Project = "geo"
  }
}

resource "aws_iam_role_policy_attachment" "batch_service_managed" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# ── Batch Job Role ─────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "batch_job_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_job" {
  name               = "geo-batch-job-role"
  assume_role_policy = data.aws_iam_policy_document.batch_job_trust.json

  tags = {
    Project = "geo"
  }
}

data "aws_iam_policy_document" "batch_job_policy" {
  statement {
    sid    = "S3Access"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      "arn:aws:s3:::${var.bucket_name}",
      "arn:aws:s3:::${var.bucket_name}/*",
    ]
  }

  statement {
    sid     = "ECRAuth"
    effect  = "Allow"
    actions = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "ECRImages"
    effect = "Allow"
    actions = [
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchCheckLayerAvailability",
    ]
    resources = [
      "arn:aws:ecr:${var.region}:${var.account_id}:repository/geo/odm",
      "arn:aws:ecr:${var.region}:${var.account_id}:repository/geo/tools",
    ]
  }

  statement {
    sid    = "CloudWatchLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = [
      "arn:aws:logs:${var.region}:${var.account_id}:log-group:/aws/batch/*",
    ]
  }
}

resource "aws_iam_role_policy" "batch_job_inline" {
  name   = "geo-batch-job-policy"
  role   = aws_iam_role.batch_job.name
  policy = data.aws_iam_policy_document.batch_job_policy.json
}

# ── EC2 Instance Role for Batch ECS Instances ──────────────────────────────────

data "aws_iam_policy_document" "batch_instance_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_instance" {
  name               = "geo-batch-instance-role"
  assume_role_policy = data.aws_iam_policy_document.batch_instance_trust.json

  tags = {
    Project = "geo"
  }
}

resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch_instance" {
  name = "geo-batch-instance-profile"
  role = aws_iam_role.batch_instance.name
}

# ── EC2 Spot Fleet Role ────────────────────────────────────────────────────────
# The AmazonEC2SpotFleetTaggingRole is pre-created in most AWS accounts.
# We create it here if it doesn't already exist; Terraform will adopt the
# existing role on apply (or create a new one if absent).

data "aws_iam_policy_document" "spot_fleet_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["spotfleet.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "spot_fleet" {
  name               = "AmazonEC2SpotFleetTaggingRole"
  assume_role_policy = data.aws_iam_policy_document.spot_fleet_trust.json

  # Lifecycle: ignore name changes so the resource adopts the existing role
  # if it already exists in the account. Apply will fail with a conflict error
  # if the role pre-exists — run `terraform import` to adopt it in that case.
  tags = {
    Project     = "geo"
    ManagedBy   = "terraform"
  }

  lifecycle {
    ignore_changes = [tags]
  }
}

resource "aws_iam_role_policy_attachment" "spot_fleet_tagging" {
  role       = aws_iam_role.spot_fleet.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}
