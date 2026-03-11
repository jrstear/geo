include "root" {
  path = find_in_parent_folders("root.hcl")
}

terraform {
  source = "${get_repo_root()}/infra/modules/s3"
}

inputs = {
  environment = "dev"
  region      = "us-west-2"
  account_id  = get_aws_account_id()
}
