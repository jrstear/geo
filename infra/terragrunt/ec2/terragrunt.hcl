# Per-job ODM EC2 launcher — Terragrunt template (geo-elmk).
#
# This file is a TEMPLATE.  Do NOT run terragrunt against it in place.
#
# Usage (typically driven by odium, but valid by hand):
#
#   1. Copy into the job directory (no substitution needed — GEO_HOME drives
#      the source path at runtime):
#        cp $GEO_HOME/infra/terragrunt/ec2/terragrunt.hcl ~/stratus/{job}/ec2/
#
#   2. Set required env vars (and any optional overrides):
#        export GEO_HOME=$HOME/git/geo            # ← required: location of the geo checkout
#        export ODM_PROJECT=bsn/{job}
#        export ODM_JOB_NAME={job}                # ← suffix on globally-named AWS resources
#        export ODM_NOTIFY_EMAIL=you@example.com
#        # optional: ODM_INSTANCE_TYPE, ODM_EBS_SIZE_GB, ODM_USE_SPOT, ODM_IMAGE,
#        #          GRAFANA_*, ODM_BUCKET, ODM_REGION, ODM_NOTIFY_PHONE, ODM_SSH_CIDR
#
#   3. Apply:
#        cd ~/stratus/{job}/ec2 && terragrunt apply
#
# Concurrent jobs:
#   Each job dir gets its own terraform state under ./terraform.tfstate, and the
#   ODM_JOB_NAME suffix keeps globally-named AWS resources from colliding (IAM
#   role, instance profile, security group, key pair, SNS topic, EventBridge
#   rules, S3 scripts prefix — see infra/ec2/main.tf locals.ws_suffix).
#
# Cleanup:
#   `terragrunt destroy` tears down the AWS resources.  After destroy succeeds,
#   `rm -rf ~/stratus/{job}/ec2/.terragrunt-cache ~/stratus/{job}/ec2/terraform.tfstate*`
#   removes local state.

terraform {
  source = "${get_env("GEO_HOME")}/infra/ec2"
}

remote_state {
  backend = "local"
  config = {
    path = "${get_terragrunt_dir()}/terraform.tfstate"
  }
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
}

inputs = {
  # ── Required: fail loud at plan time if any of these are missing ─────────────
  project      = get_env("ODM_PROJECT")
  job_name     = get_env("ODM_JOB_NAME")
  notify_email = get_env("ODM_NOTIFY_EMAIL")

  # ── Optional EC2 / ODM overrides ─────────────────────────────────────────────
  bucket_name   = get_env("ODM_BUCKET", "stratus-jrstear")
  region        = get_env("ODM_REGION", "us-west-2")
  instance_type = get_env("ODM_INSTANCE_TYPE", "r5.4xlarge")
  ebs_size_gb   = tonumber(get_env("ODM_EBS_SIZE_GB", "500"))
  use_spot      = tobool(get_env("ODM_USE_SPOT", "false"))
  odm_image     = get_env("ODM_IMAGE", "opendronemap/odm:3.5.6")
  notify_phone  = get_env("ODM_NOTIFY_PHONE", "")
  ssh_cidr      = get_env("ODM_SSH_CIDR", "0.0.0.0/0")

  # ── Grafana Cloud telemetry (optional; monitoring skipped when empty) ────────
  grafana_prom_url  = get_env("GRAFANA_PROM_URL", "")
  grafana_prom_user = get_env("GRAFANA_PROM_USER", "")
  grafana_loki_url  = get_env("GRAFANA_LOKI_URL", "")
  grafana_loki_user = get_env("GRAFANA_LOKI_USER", "")
  grafana_api_key   = get_env("GRAFANA_API_KEY", "")
  grafana_sa_key    = get_env("GRAFANA_SA_KEY", "")
  grafana_stack_url = get_env("GRAFANA_STACK_URL", "")
}
