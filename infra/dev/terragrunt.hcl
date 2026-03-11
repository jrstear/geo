include "root" {
  path = find_in_parent_folders("root.hcl")
}

inputs = {
  environment = "dev"
  region      = "us-east-1"
  account_id  = get_aws_account_id()
}
