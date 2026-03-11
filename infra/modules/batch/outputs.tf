output "queue_arn" {
  description = "ARN of the geo-odm-queue job queue"
  value       = aws_batch_job_queue.odm_queue.arn
}

output "odm_job_def_arn" {
  description = "ARN of the odm-run job definition"
  value       = aws_batch_job_definition.odm_run.arn
}

output "rmse_job_def_arn" {
  description = "ARN of the rmse-calc job definition"
  value       = aws_batch_job_definition.rmse_calc.arn
}

output "compute_env_arn" {
  description = "ARN of the geo-odm-compute compute environment"
  value       = aws_batch_compute_environment.odm_spot.arn
}
