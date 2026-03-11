output "job_role_arn" {
  description = "ARN of the geo-batch-job-role"
  value       = aws_iam_role.batch_job.arn
}

output "service_role_arn" {
  description = "ARN of the geo-batch-service-role"
  value       = aws_iam_role.batch_service.arn
}

output "instance_profile_arn" {
  description = "ARN of the geo-batch-instance-profile"
  value       = aws_iam_instance_profile.batch_instance.arn
}

output "spot_fleet_role_arn" {
  description = "ARN of the AmazonEC2SpotFleetTaggingRole"
  value       = aws_iam_role.spot_fleet.arn
}
