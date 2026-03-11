include "root" {
  path = find_in_parent_folders("root.hcl")
}

inputs = {
  environment = "dev"
  region      = "us-west-2"
  account_id  = get_aws_account_id()
}
