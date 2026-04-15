# Spot EC2 instance for ODM reconstruction runs — fully automated pipeline.
#
# ── USAGE (readable any time; see also: terraform output usage after apply) ────
#
# S3 layout (mirrors /data/project/ on the instance):
#   s3://BUCKET/PROJECT/
#     images/         ← you upload raw images here before apply
#     gcp_list.txt    ← you upload GCP file here before apply
#     opensfm/ odm_meshing/ odm_orthophoto/ odm_dem/ odm_report/ ...
#                     ← written automatically on pipeline completion
#
# STEP 1 — Upload data to S3:
#   aws s3 sync /path/to/images/ s3://BUCKET/PROJECT/images/ \
#     --exclude "*.MRK" --exclude "*.nav" --exclude "*.obs" --exclude "*.bin" \
#     --region us-west-2
#   aws s3 cp ~/stratus/{job}/gcp_list.txt s3://BUCKET/PROJECT/gcp_list.txt \
#     --region us-west-2
#
# STEP 2 — Apply (creates spot instance; pipeline starts automatically):
#   terraform apply \
#     -var="project=bsn/aztec" \
#     -var="notify_email=you@example.com"          # free; confirm subscription email
#     [-var="notify_phone=+15055551234"]            # optional SMS, ~$0.006/message
#
#   After apply, AWS sends a "Subscription Confirmation" email — click the link
#   or notifications will be silently dropped. SMS is auto-confirmed.
#
# STEP 3 — Save SSH key (first time only, after apply):
#   terraform output -raw private_key_pem > ~/.ssh/geo-odm-ec2.pem && chmod 600 ~/.ssh/geo-odm-ec2.pem
#
# STEP 4 — Test SNS (after apply, before pipeline starts):
#   aws sns publish \
#     --topic-arn $(terraform output -raw sns_topic_arn) \
#     --subject "ODM test" --message "test — if you see this, notifications work." \
#     --region us-west-2
#
# STEP 5 — Watch logs (optional):
#   ssh -i ~/.ssh/geo-odm-ec2.pem ec2-user@$(terraform output -raw public_ip)
#   tail -f /var/log/odm-bootstrap.log    # full pipeline + S3 sync output
#
# STEP 6 — Download deliverables (after "complete" SNS):
#   aws s3 sync s3://BUCKET/PROJECT/odm_orthophoto/ ./odm_orthophoto/ --profile personal
#   aws s3 sync s3://BUCKET/PROJECT/odm_dem/        ./odm_dem/        --profile personal
#   aws s3 sync s3://BUCKET/PROJECT/odm_report/     ./odm_report/     --profile personal
#
# STEP 7 — Tear down:
#   terraform destroy   # cancels persistent spot request + deletes EBS
#
# SPOT INTERRUPTION: AWS auto-restarts the instance (persistent spot) and
#   odm-bootstrap.sh resumes the pipeline from the last completed stage.
#   No action needed. See 'terraform output instructions' for restart command.
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

variable "bucket_name" {
  description = "S3 bucket containing flight data"
  default     = "stratus-jrstear"
}

variable "project" {
  description = <<-EOD
    S3 path prefix for this job, e.g. bsn/aztec or bsn/red-rocks.
    All job data lives under s3://BUCKET/PROJECT/:
      images/      ← upload raw images here before apply
      gcp_list.txt ← upload GCP file here before apply
      opensfm/ odm_orthophoto/ etc. ← written on completion
  EOD
  default = "PROJECT"  # always override: -var="project=bsn/sitename"
}

variable "instance_type" {
  description = "EC2 instance type."
  default     = "r5.4xlarge"
  # Spot pricing in us-west-2 (~70% discount vs on-demand):
  #   r5.4xlarge  ~$0.24/hr  16 vCPU  128 GB  ← default; good for medium quality
  #   r5.8xlarge  ~$0.48/hr  32 vCPU  256 GB  ← high quality / large datasets
  #   m5.4xlarge  ~$0.18/hr  16 vCPU   64 GB  ← lighter jobs
}

variable "use_spot" {
  description = "Use spot instances (cheaper but interruptible). Set false for on-demand."
  type        = bool
  default     = false  # default to on-demand — spot interruptions cause costly rework
}

variable "spot_max_price" {
  description = "Max spot bid in USD/hr. Empty = on-demand price cap (recommended). Only applies when use_spot=true."
  default     = ""
}

variable "ebs_size_gb" {
  description = "EBS root volume size in GB. Aztec scale: 400 sufficient; Red Rocks: 600+."
  default     = 500
}

variable "notify_email" {
  description = "Email for per-stage completion alerts. Empty = no email subscription."
  default     = ""
}

variable "notify_phone" {
  description = "Phone (E.164, e.g. +15055551234) for SMS alerts. Empty = disabled."
  default     = ""
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH in."
  default     = "0.0.0.0/0"
}

variable "odm_image" {
  description = <<-EOD
    ODM Docker image to pull and run.
    Default: stock opendronemap/odm:3.5.6.
    The exifread DJI MakerNote bug is patched at boot by odm-bootstrap.sh
    (no custom image build needed).
  EOD
  default     = "opendronemap/odm:3.5.6"
}

# ── Grafana Cloud telemetry (all optional; monitoring is skipped when empty) ───

variable "grafana_prom_url" {
  description = "Grafana Cloud Prometheus remote_write URL (e.g. https://prometheus-prod-XX-XXX.grafana.net/api/prom/push)"
  default     = ""
}

variable "grafana_prom_user" {
  description = "Grafana Cloud Prometheus numeric stack ID (basic-auth username for remote_write)"
  default     = ""
}

variable "grafana_loki_url" {
  description = "Grafana Cloud Loki push URL (e.g. https://logs-prod-XXX.grafana.net/loki/api/v1/push)"
  default     = ""
}

variable "grafana_loki_user" {
  description = "Grafana Cloud Loki numeric user ID (basic-auth username for Loki push)"
  default     = ""
}

variable "grafana_api_key" {
  description = "Grafana Cloud API key / service account token (needs metrics:write, logs:write, annotations:write)"
  default     = ""
  sensitive   = true
}

variable "grafana_sa_key" {
  description = "Grafana Cloud service account token (glsa_...) for annotations. Separate from the data-plane API key."
  default     = ""
  sensitive   = true
}

variable "grafana_stack_url" {
  description = "Grafana Cloud stack base URL for annotations API (e.g. https://yourorg.grafana.net)"
  default     = ""
}

# ── Locals ─────────────────────────────────────────────────────────────────────

locals {
  s3_base   = "s3://${var.bucket_name}/${var.project}"
  s3_images = "${local.s3_base}/images"
  s3_gcp    = "${local.s3_base}/gcp_list.txt"

  # --optimize-disk-space intentionally excluded: it deletes undistorted images
  # after openmvs/texturing, preventing resume from orthophoto after interruption.
  # --max-concurrency is set per-stage by odm-run.sh.
  odm_flags = "--pc-quality medium --feature-quality high --orthophoto-resolution 5 --dtm --dsm --dem-resolution 5 --cog --build-overviews"

  scripts_s3_prefix = "odm-scripts"
}

# ── SSH Key ────────────────────────────────────────────────────────────────────

resource "tls_private_key" "odm" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "odm" {
  key_name   = "geo-odm-ec2"
  public_key = tls_private_key.odm.public_key_openssh
  tags       = { Project = "geo" }
}

# ── IAM: S3 read/write + SNS publish ──────────────────────────────────────────

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "odm" {
  name               = "geo-odm-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = { Project = "geo" }
}

data "aws_iam_policy_document" "s3_access" {
  statement {
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [
      "arn:aws:s3:::${var.bucket_name}",
      "arn:aws:s3:::${var.bucket_name}/*",
    ]
  }
}

resource "aws_iam_role_policy" "odm_s3" {
  name   = "geo-odm-s3"
  role   = aws_iam_role.odm.name
  policy = data.aws_iam_policy_document.s3_access.json
}

data "aws_iam_policy_document" "sns_publish" {
  statement {
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.odm_alerts.arn]
  }
}

resource "aws_iam_role_policy" "odm_sns" {
  name   = "geo-odm-sns"
  role   = aws_iam_role.odm.name
  policy = data.aws_iam_policy_document.sns_publish.json
}

data "aws_iam_policy_document" "ec2_spot" {
  statement {
    effect = "Allow"
    actions = [
      "ec2:CancelSpotInstanceRequests",
      "ec2:DescribeSpotInstanceRequests",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "odm_ec2_spot" {
  name   = "geo-odm-ec2-spot"
  role   = aws_iam_role.odm.name
  policy = data.aws_iam_policy_document.ec2_spot.json
}

data "aws_iam_policy_document" "pricing" {
  statement {
    effect    = "Allow"
    actions   = ["pricing:GetProducts"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "odm_pricing" {
  name   = "geo-odm-pricing"
  role   = aws_iam_role.odm.name
  policy = data.aws_iam_policy_document.pricing.json
}

data "aws_iam_policy_document" "ecr_pull" {
  # GetAuthorizationToken must be on "*"
  statement {
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  # Image pull actions scoped to our ECR registry
  statement {
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = ["arn:aws:ecr:${var.region}:*:repository/*"]
  }
}

resource "aws_iam_role_policy" "odm_ecr" {
  name   = "geo-odm-ecr-pull"
  role   = aws_iam_role.odm.name
  policy = data.aws_iam_policy_document.ecr_pull.json
}

resource "aws_iam_instance_profile" "odm" {
  name = "geo-odm-ec2-profile"
  role = aws_iam_role.odm.name
}

# ── Security group ─────────────────────────────────────────────────────────────

resource "aws_security_group" "odm" {
  name        = "geo-odm-ec2-sg"
  description = "SSH ingress + all egress for ODM EC2 run"

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

  tags = { Project = "geo" }
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

resource "aws_sns_topic" "odm_alerts" {
  name = "geo-odm-alerts"
  tags = { Project = "geo" }
}

data "aws_iam_policy_document" "sns_topic_policy" {
  statement {
    sid     = "AllowEventBridge"
    effect  = "Allow"
    actions = ["SNS:Publish"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
    resources = [aws_sns_topic.odm_alerts.arn]
  }
  statement {
    sid     = "AllowInstanceRole"
    effect  = "Allow"
    actions = ["SNS:Publish"]
    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.odm.arn]
    }
    resources = [aws_sns_topic.odm_alerts.arn]
  }
}

resource "aws_sns_topic_policy" "odm_alerts" {
  arn    = aws_sns_topic.odm_alerts.arn
  policy = data.aws_iam_policy_document.sns_topic_policy.json
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.notify_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.odm_alerts.arn
  protocol  = "email"
  endpoint  = var.notify_email
}

resource "aws_sns_topic_subscription" "sms" {
  count     = var.notify_phone != "" ? 1 : 0
  topic_arn = aws_sns_topic.odm_alerts.arn
  protocol  = "sms"
  endpoint  = var.notify_phone
}

# ── EventBridge alerts ─────────────────────────────────────────────────────────

resource "aws_cloudwatch_event_rule" "spot_interruption" {
  name        = "geo-odm-spot-interruption"
  description = "Spot interruption warning for geo-odm instance"
  event_pattern = jsonencode({
    source        = ["aws.ec2"]
    "detail-type" = ["EC2 Spot Instance Interruption Warning"]
    detail        = { "instance-id" = [aws_instance.odm.id] }
  })
  tags = { Project = "geo" }
}

resource "aws_cloudwatch_event_target" "spot_interruption_sns" {
  rule      = aws_cloudwatch_event_rule.spot_interruption.name
  target_id = "SNS"
  arn       = aws_sns_topic.odm_alerts.arn
}

resource "aws_cloudwatch_event_rule" "instance_state" {
  name        = "geo-odm-instance-state"
  description = "geo-odm instance stopped or terminated"
  event_pattern = jsonencode({
    source        = ["aws.ec2"]
    "detail-type" = ["EC2 Instance State-change Notification"]
    detail = {
      state         = ["stopped", "terminated"]
      "instance-id" = [aws_instance.odm.id]
    }
  })
  tags = { Project = "geo" }
}

resource "aws_cloudwatch_event_target" "instance_state_sns" {
  rule      = aws_cloudwatch_event_rule.instance_state.name
  target_id = "SNS"
  arn       = aws_sns_topic.odm_alerts.arn
}

# ── Scripts in S3 ─────────────────────────────────────────────────────────────
# Uploaded from infra/ec2/scripts/; user_data downloads them at boot.
# To update a script without terraform apply:
#   aws s3 cp infra/ec2/scripts/odm-run.sh s3://BUCKET/odm-scripts/ --profile personal
#   ssh ec2-user@IP 'sudo aws s3 cp s3://BUCKET/odm-scripts/odm-run.sh /usr/local/bin/ --region us-west-2'

resource "aws_s3_object" "odm_run" {
  bucket = var.bucket_name
  key    = "${local.scripts_s3_prefix}/odm-run.sh"
  source = "${path.module}/scripts/odm-run.sh"
  etag   = filemd5("${path.module}/scripts/odm-run.sh")
}

resource "aws_s3_object" "odm_bootstrap" {
  bucket = var.bucket_name
  key    = "${local.scripts_s3_prefix}/odm-bootstrap.sh"
  source = "${path.module}/scripts/odm-bootstrap.sh"
  etag   = filemd5("${path.module}/scripts/odm-bootstrap.sh")
}

resource "aws_s3_object" "spot_watcher" {
  bucket = var.bucket_name
  key    = "${local.scripts_s3_prefix}/spot-watcher.sh"
  source = "${path.module}/scripts/spot-watcher.sh"
  etag   = filemd5("${path.module}/scripts/spot-watcher.sh")
}

resource "aws_s3_object" "odm_monitor" {
  bucket = var.bucket_name
  key    = "${local.scripts_s3_prefix}/odm-monitor.sh"
  source = "${path.module}/scripts/odm-monitor.sh"
  etag   = filemd5("${path.module}/scripts/odm-monitor.sh")
}

resource "aws_s3_object" "odm_progress" {
  bucket = var.bucket_name
  key    = "${local.scripts_s3_prefix}/odm-progress.sh"
  source = "${path.module}/scripts/odm-progress.sh"
  etag   = filemd5("${path.module}/scripts/odm-progress.sh")
}

resource "aws_s3_object" "true_ortho" {
  bucket = var.bucket_name
  key    = "${local.scripts_s3_prefix}/true_ortho.py"
  source = "${path.module}/../../experimental/true_ortho.py"
  etag   = filemd5("${path.module}/../../experimental/true_ortho.py")
}

# ── EC2 Spot instance ──────────────────────────────────────────────────────────

resource "aws_instance" "odm" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.odm.name
  key_name               = aws_key_pair.odm.key_name
  vpc_security_group_ids = [aws_security_group.odm.id]

  # Spot: persistent + stop — EBS survives interruption, auto-restarts.
  # On-demand: no interruptions, ~70% more expensive but no rework.
  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price                      = var.spot_max_price != "" ? var.spot_max_price : null
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
      }
    }
  }

  root_block_device {
    volume_size           = var.ebs_size_gb  # 300 caused disk-full at medium quality; default 500
    volume_type           = "gp3"
    throughput            = 250
    iops                  = 3000
    delete_on_termination = true
  }

  user_data = base64encode(<<-USERDATA
    #!/bin/bash
    set -euo pipefail
    exec > /var/log/user-data.log 2>&1

    dnf update -y
    dnf install -y docker screen htop cronie
    systemctl enable --now docker
    systemctl enable --now crond
    usermod -aG docker ec2-user

    # ECR login (no-op if using Docker Hub image; harmless either way).
    aws ecr get-login-password --region "${var.region}" \
      | docker login --username AWS --password-stdin \
          "$(echo "${var.odm_image}" | cut -d/ -f1)" 2>/dev/null || true

    # Pull ODM image in background (~4–6 GB); odm-bootstrap.sh waits for it.
    docker pull "${var.odm_image}" &

    # Runtime config — written once; scripts source this on every run.
    cat > /etc/odm-env << 'ENVFILE'
    export BUCKET="${var.bucket_name}"
    export PROJECT="${var.project}"
    export REGION="${var.region}"
    export SNS_TOPIC="${aws_sns_topic.odm_alerts.arn}"
    export ODM_FLAGS="${local.odm_flags}"
    export ODM_IMAGE="${var.odm_image}"
    export GRAFANA_PROM_URL="${var.grafana_prom_url}"
    export GRAFANA_PROM_USER="${var.grafana_prom_user}"
    export GRAFANA_LOKI_URL="${var.grafana_loki_url}"
    export GRAFANA_LOKI_USER="${var.grafana_loki_user}"
    export GRAFANA_API_KEY="${var.grafana_api_key}"
    export GRAFANA_STACK_URL="${var.grafana_stack_url}"
    export GRAFANA_SA_KEY="${var.grafana_sa_key}"
    ENVFILE

    # Download scripts from S3 (uploaded by Terraform from infra/ec2/scripts/).
    # To update without terraform apply:
    #   aws s3 cp infra/ec2/scripts/odm-run.sh s3://${var.bucket_name}/${local.scripts_s3_prefix}/ --profile personal
    #   ssh ec2-user@IP 'sudo aws s3 cp s3://${var.bucket_name}/${local.scripts_s3_prefix}/odm-run.sh /usr/local/bin/ --region ${var.region}'
    for script in odm-run.sh odm-bootstrap.sh spot-watcher.sh odm-monitor.sh odm-progress.sh true_ortho.py; do
      aws s3 cp "s3://${var.bucket_name}/${local.scripts_s3_prefix}/$script" \
        "/usr/local/bin/$script" --region "${var.region}"
    done
    chmod +x /usr/local/bin/odm-run.sh /usr/local/bin/odm-bootstrap.sh \
             /usr/local/bin/spot-watcher.sh /usr/local/bin/odm-monitor.sh \
             /usr/local/bin/odm-progress.sh

    # Install telemetry stack (no-op if GRAFANA_API_KEY is empty).
    # Runs synchronously so metrics are flowing before ODM starts.
    /usr/local/bin/odm-monitor.sh >> /var/log/odm-monitor.log 2>&1 || true

    # @reboot cron fires odm-bootstrap.sh on every boot — enables spot resume.
    echo "@reboot root /usr/local/bin/odm-bootstrap.sh" > /etc/cron.d/odm-bootstrap
    chmod 644 /etc/cron.d/odm-bootstrap

    nohup /usr/local/bin/spot-watcher.sh >> /var/log/spot-watcher.log 2>&1 &
    nohup /usr/local/bin/odm-bootstrap.sh &

    echo "user-data complete" | tee /var/log/user-data-done
  USERDATA
  )

  depends_on = [
    aws_s3_object.odm_run,
    aws_s3_object.odm_bootstrap,
    aws_s3_object.spot_watcher,
    aws_s3_object.odm_monitor,
  ]

  tags = {
    Project    = "geo"
    Purpose    = "odm-run"
    Name       = "geo-odm-ec2"
    ODMProject = var.project
  }
}

# ── Outputs ────────────────────────────────────────────────────────────────────

output "public_ip" {
  value = aws_instance.odm.public_ip
}

output "instance_id" {
  value = aws_instance.odm.id
}

output "sns_topic_arn" {
  value       = aws_sns_topic.odm_alerts.arn
  description = "SNS topic for pipeline alerts"
}

output "private_key_pem" {
  value       = tls_private_key.odm.private_key_pem
  sensitive   = true
  description = "Save with: terraform output -raw private_key_pem > ~/.ssh/geo-odm-ec2.pem && chmod 600 ~/.ssh/geo-odm-ec2.pem"
}

# usage output: only var.* and local.* — available before and after apply.
output "usage" {
  value = <<-EOT

    ═══════════════════════════════════════════════════════════════════════════════
     ODM Spot Pipeline — project: ${var.project}   instance: ${var.instance_type}
     s3://${var.bucket_name}/${var.project}/
    ═══════════════════════════════════════════════════════════════════════════════

    ── STEP 1: UPLOAD DATA TO S3 ─────────────────────────────────────────────────

    # Raw images (from drone SD card):
    aws s3 sync /path/to/images/ ${local.s3_images}/ \
      --exclude "*.MRK" --exclude "*.nav" --exclude "*.obs" --exclude "*.bin" \
      --region ${var.region}

    # GCP file:
    aws s3 cp ~/stratus/${var.project}/gcp_list.txt \
      ${local.s3_gcp} \
      --region ${var.region}

    ── STEP 2: APPLY ─────────────────────────────────────────────────────────────

    # Email only (free):
    terraform apply \
      -var="project=${var.project}" \
      -var="notify_email=you@example.com"

    # Email + SMS (~$0.006/message, ~$0.06/run for 10 stages):
    terraform apply \
      -var="project=${var.project}" \
      -var="notify_email=you@example.com" \
      -var="notify_phone=+15055551234"

    # After apply, AWS sends a "Subscription Confirmation" email — click the link
    # or SNS messages will be silently dropped. SMS subscriptions are auto-confirmed.

    # Then everything runs automatically:
    #   1. Instance syncs project from S3 → EBS (images + any prior stage outputs)
    #   2. Runs ODM stages with tuned concurrency (16/8/4 threads per stage)
    #   3. SNS notification on each stage completion and on failure
    #   4. Syncs outputs back to s3://${var.bucket_name}/${var.project}/
    #   5. SNS: "complete — run terraform destroy"

    ── STEP 3: SSH KEY (first time only, after apply) ────────────────────────────

    terraform output -raw private_key_pem > ~/.ssh/geo-odm-ec2.pem
    chmod 600 ~/.ssh/geo-odm-ec2.pem

    ── TEST SNS (after apply, before pipeline starts) ────────────────────────────

    aws sns publish \
      --topic-arn $(terraform output -raw sns_topic_arn) \
      --subject "ODM test" \
      --message "SNS test from geo-odm — if you see this, notifications are working." \
      --region ${var.region}

    ── WATCH LOGS (optional) ─────────────────────────────────────────────────────

    ssh -i ~/.ssh/geo-odm-ec2.pem ec2-user@${aws_instance.odm.public_ip}
    tail -f /var/log/odm-bootstrap.log    # full pipeline output + S3 sync
    tail -f /var/log/spot-watcher.log     # interruption watcher

    ── SPOT INTERRUPTION ─────────────────────────────────────────────────────────

    # AWS auto-restarts (persistent spot) and resumes automatically. No action needed.
    # To force a manual restart: aws ec2 start-instances --instance-ids <ID> --profile personal

    ── DOWNLOAD DELIVERABLES ─────────────────────────────────────────────────────

    aws s3 sync s3://${var.bucket_name}/${var.project}/odm_orthophoto/ ./odm_orthophoto/ --profile personal
    aws s3 sync s3://${var.bucket_name}/${var.project}/odm_dem/        ./odm_dem/        --profile personal
    aws s3 sync s3://${var.bucket_name}/${var.project}/odm_report/     ./odm_report/     --profile personal

    ── TEAR DOWN ─────────────────────────────────────────────────────────────────

    terraform destroy   # cancels persistent spot request + deletes EBS

  EOT
}

# instructions output: post-apply details with instance-specific values.
output "instructions" {
  value = <<-EOT

    ── INSTANCE ──────────────────────────────────────────────────────────────────

    Public IP:   ${aws_instance.odm.public_ip}
    Instance ID: ${aws_instance.odm.id}
    SNS topic:   ${aws_sns_topic.odm_alerts.arn}

    ── SSH ───────────────────────────────────────────────────────────────────────

    ssh -i ~/.ssh/geo-odm-ec2.pem ec2-user@${aws_instance.odm.public_ip}

    ── TEST SNS ──────────────────────────────────────────────────────────────────

    aws sns publish \
      --topic-arn ${aws_sns_topic.odm_alerts.arn} \
      --subject "ODM test" --message "test" \
      --region ${var.region}

    ── MANUAL RESTART (after spot interruption) ──────────────────────────────────

    aws ec2 start-instances --instance-ids ${aws_instance.odm.id} \
      --region ${var.region}

  EOT
}
