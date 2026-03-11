include "root" {
  path = find_in_parent_folders("root.hcl")
}

terraform {
  source = "${get_repo_root()}/infra/modules/ecr"
}

inputs = {
  environment = "prod"
  region      = "us-east-1"
  account_id  = get_aws_account_id()
}
