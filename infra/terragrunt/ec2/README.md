# Per-job ODM EC2 launcher (Terragrunt template)

`terragrunt.hcl` here is a template that gets copied into each job's
directory (e.g. `~/stratus/aztec13/ec2/`) so multiple ODM jobs can run
concurrently without resource-name collisions or state-file overlap.

## How it works

1. **Source pinning** — `terraform { source = "${get_env("GEO_HOME")}/infra/ec2" }`
   points at the geo checkout's terraform module at runtime. `GEO_HOME` is a
   required env var, normally `$HOME/git/geo`. No file-copy substitution
   needed — the same template file works on any machine that exports
   `GEO_HOME`.

2. **Per-job state** — local backend at `${job_dir}/ec2/terraform.tfstate`.
   Terragrunt also gives `.terragrunt-cache/` for free per-dir isolation.

3. **Per-job resource names** — `ODM_JOB_NAME` env var → `var.job_name`
   (required, validated non-empty) → `local.ws_suffix = "-${var.job_name}"`
   in `infra/ec2/main.tf`, applied as a suffix to:

   - `aws_key_pair.odm` → `geo-odm-ec2-{job_name}`
   - `aws_iam_role.odm` → `geo-odm-ec2-role-{job_name}`
   - `aws_iam_instance_profile.odm` → `geo-odm-ec2-profile-{job_name}`
   - `aws_security_group.odm` → `geo-odm-ec2-sg-{job_name}`
   - `aws_sns_topic.odm_alerts` → `geo-odm-alerts-{job_name}`
   - `aws_cloudwatch_event_rule.spot_interruption` → `geo-odm-spot-interruption-{job_name}`
   - `aws_cloudwatch_event_rule.instance_state` → `geo-odm-instance-state-{job_name}`
   - `local.scripts_s3_prefix` → `odm-scripts-{job_name}/` (under `s3://${var.bucket_name}/`)

   `job_name` is required — direct `terraform apply` against `infra/ec2/`
   must pass `-var="job_name=..."` explicitly.

## Manual usage

```bash
JOB=aztec13
mkdir -p ~/stratus/${JOB}/ec2
cp $GEO_HOME/infra/terragrunt/ec2/terragrunt.hcl ~/stratus/${JOB}/ec2/

cd ~/stratus/${JOB}/ec2
export GEO_HOME=$HOME/git/geo            # required
export ODM_PROJECT=bsn/${JOB}
export ODM_JOB_NAME=${JOB}
export ODM_NOTIFY_EMAIL=jrstear@gmail.com
export ODM_IMAGE=658302145097.dkr.ecr.us-west-2.amazonaws.com/odm:3.6.0-patched-pr48-pr2008
# ... any other ODM_* / GRAFANA_* env vars

terragrunt apply
```

## Cleanup

```bash
cd ~/stratus/${JOB}/ec2 && terragrunt destroy
rm -rf .terragrunt-cache terraform.tfstate*
```

odium auto-cleans `.terragrunt-cache/` and `terraform.tfstate*` after a
successful destroy.

## Cross-references

- `infra/ec2/main.tf` — the source module (`var.job_name`, `local.ws_suffix`)
- bead `geo-elmk` — design + sequencing decisions
- bead `geo-8fg` — stage-switching adds 4 more globally-named resources that
  will follow the same suffix pattern when that lands
