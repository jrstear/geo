include "root" {
  path = find_in_parent_folders("root.hcl")
}

terraform {
  source = "${get_repo_root()}/infra/modules/batch"
}

dependency "iam" {
  config_path = "../iam"
  mock_outputs = {
    job_role_arn          = "arn:aws:iam::869054869504:role/geo-batch-job-role"
    service_role_arn      = "arn:aws:iam::869054869504:role/geo-batch-service-role"
    instance_profile_arn  = "arn:aws:iam::869054869504:instance-profile/geo-batch-instance-profile"
    spot_fleet_role_arn   = "arn:aws:iam::869054869504:role/AmazonEC2SpotFleetTaggingRole"
  }
  mock_outputs_allowed_terraform_commands = ["validate", "plan"]
}

dependency "ecr" {
  config_path = "../ecr"
  mock_outputs = {
    ecr_base       = "869054869504.dkr.ecr.us-east-1.amazonaws.com"
    odm_repo_url   = "869054869504.dkr.ecr.us-east-1.amazonaws.com/geo/odm"
    tools_repo_url = "869054869504.dkr.ecr.us-east-1.amazonaws.com/geo/tools"
  }
  mock_outputs_allowed_terraform_commands = ["validate", "plan"]
}

inputs = {
  environment          = "dev"
  region               = "us-east-1"
  account_id           = get_aws_account_id()
  job_role_arn         = dependency.iam.outputs.job_role_arn
  service_role_arn     = dependency.iam.outputs.service_role_arn
  instance_profile_arn = dependency.iam.outputs.instance_profile_arn
  spot_fleet_role_arn  = dependency.iam.outputs.spot_fleet_role_arn
  odm_image_url        = "${dependency.ecr.outputs.ecr_base}/geo/odm:latest"
  tools_image_url      = "${dependency.ecr.outputs.ecr_base}/geo/tools:latest"
}
