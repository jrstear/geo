variable "environment" {
  description = "Deployment environment (dev or prod)"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "account_id" {
  description = "AWS account ID"
  type        = string
  default     = ""
}

variable "job_role_arn" {
  description = "ARN of the IAM role used by Batch job containers"
  type        = string
}

variable "service_role_arn" {
  description = "ARN of the IAM role used by the Batch service"
  type        = string
}

variable "instance_profile_arn" {
  description = "ARN of the EC2 instance profile for Batch compute instances"
  type        = string
}

variable "spot_fleet_role_arn" {
  description = "ARN of the EC2 Spot Fleet IAM role"
  type        = string
}

variable "odm_image_url" {
  description = "Full ECR image URL for the ODM container (e.g. 123456789.dkr.ecr.us-west-2.amazonaws.com/geo/odm:latest)"
  type        = string
}

variable "tools_image_url" {
  description = "Full ECR image URL for the tools container (e.g. 123456789.dkr.ecr.us-west-2.amazonaws.com/geo/tools:latest)"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for the Batch compute environment"
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "List of security group IDs for the Batch compute environment"
  type        = list(string)
  default     = []
}
