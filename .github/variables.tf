variable "projectid" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "asia-southeast1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "asia-southeast1-a"
}

variable "instance_name" {
  description = "GCP Compute Engine Instance"
  type        = string
}

variable "email" {
  description = "Email"
  type        = string
  sensitive   = true
}