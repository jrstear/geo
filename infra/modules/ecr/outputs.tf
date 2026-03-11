output "odm_repo_url" {
  description = "URL of the geo/odm ECR repository"
  value       = aws_ecr_repository.odm.repository_url
}

output "tools_repo_url" {
  description = "URL of the geo/tools ECR repository"
  value       = aws_ecr_repository.tools.repository_url
}

output "ecr_base" {
  description = "ECR registry base URL (account.dkr.ecr.region.amazonaws.com)"
  value       = "${var.account_id}.dkr.ecr.${var.region}.amazonaws.com"
}
