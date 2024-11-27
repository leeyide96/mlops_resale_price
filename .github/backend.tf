terraform {
  backend "gcs" {
    bucket = "your-own-bucket-terraform-state"
    prefix = "terraform/state"
  }
}