provider "google" {
  project = var.projectid
  region  = var.region
}

resource "google_storage_bucket" "airflow_bucket" {
  name                        = "your-own-bucket"
  location                    = var.region
  uniform_bucket_level_access = true
}

resource "google_storage_bucket_object" "functions_info" {
  for_each = fileset("../functions", "**/*")

  name   = each.value
  bucket = google_storage_bucket.airflow_bucket.name
  source = "../functions/${each.value}"
}

data "external" "list_functions" {
  program = ["bash", "-c", "cd ../functions && find */ -type d -exec basename {} \\;| sort| uniq | jq -R -s '{ \"folder_names\": split(\"\\n\")[:-1] | join(\",\") }'"]
}

locals {
  function_folder_names = split(",", data.external.list_functions.result["folder_names"])
}

resource "google_cloudfunctions2_function" "functions" {
  for_each = { for key in local.function_folder_names : google_storage_bucket_object.functions_info["${key}.zip"].name => key }

  name     = each.value
  location = var.region
  build_config {
    runtime     = "python310"
    entry_point = "main"
    source {
      storage_source {
        bucket = google_storage_bucket.airflow_bucket.name
        object = google_storage_bucket_object.functions_info["${each.value}.zip"].name
      }
    }
  }
  service_config {
    max_instance_count = 8
    available_memory   = "512M"
  }
  lifecycle {
    replace_triggered_by = [google_storage_bucket_object.functions_info[each.key]]
  }
}

resource "google_cloud_run_service_iam_binding" "binding" {
  for_each = toset(local.function_folder_names)

  project  = google_cloudfunctions2_function.functions["${each.value}.zip"].project
  location = google_cloudfunctions2_function.functions["${each.value}.zip"].location
  service  = google_cloudfunctions2_function.functions["${each.value}.zip"].name
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}

resource "google_compute_instance" "airflow_instance" {
  name                      = var.instance_name
  machine_type              = "n2d-standard-2"
  zone                      = var.zone
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = 50
    }
  }

  network_interface {
    network = "default"
    access_config {
    }
  }

  service_account {
    email  = var.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    google-logging-enabled = "true"
    startup-script         = <<-EOF
      #!/bin/bash
      # Create directories for Docker volumes
      mkdir -p /home/airflow/dags
      chmod 755 /home/airflow/dags
    EOF
  }

  tags = ["airflow", "http-server"]
}

resource "google_compute_firewall" "airflow_firewall" {
  name    = "allow-airflow-webserver"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server"]
}

output "bucket_name" {
  value = google_storage_bucket.airflow_bucket.name
}

