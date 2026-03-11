output "bucket_name" {
  description = "Name of the ODM S3 bucket"
  value       = aws_s3_bucket.odm.bucket
}

output "bucket_arn" {
  description = "ARN of the ODM S3 bucket"
  value       = aws_s3_bucket.odm.arn
}
