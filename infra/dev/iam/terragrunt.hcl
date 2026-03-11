include "root" {
  path = find_in_parent_folders("root.hcl")
}

terraform {
  source = "${get_repo_root()}/infra/modules/iam"
}

dependency "s3" {
  config_path = "../s3"
  mock_outputs = {
    bucket_name = "geo-odm-869054869504"
  }
  mock_outputs_allowed_terraform_commands = ["validate", "plan"]
}

inputs = {
  environment = "dev"
  region      = "us-west-2"
  account_id  = get_aws_account_id()
  bucket_name = dependency.s3.outputs.bucket_name
}
