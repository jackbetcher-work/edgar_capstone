# ------------------------------------------------------------------------------
# Global External HTTPS Application Load Balancer fronting the IAP backend
# ------------------------------------------------------------------------------
# Cloud Run ingress is INTERNAL_LOAD_BALANCER, so this load balancer is the only
# public entry point. Traffic flow:
#   client -> forwarding rule (443) -> HTTPS proxy (managed cert) -> URL map
#          -> IAP-enabled backend service -> serverless NEG -> Cloud Run
# Port 80 is redirected to HTTPS.

locals {
  enable_https_lb = (
    var.enable_https_load_balancer
    && var.enable_iap_backend_service
    && var.enable_cloud_run_service
  )

  # Use the supplied domain or fall back to <ip>.nip.io, which resolves to the
  # load balancer IP without any DNS setup (fine for sandbox environments).
  lb_domain = (
    local.enable_https_lb
    ? (
      trimspace(var.lb_domain) != ""
      ? trimspace(var.lb_domain)
      : "${google_compute_global_address.sec_edgar_lb_ip[0].address}.nip.io"
    )
    : null
  )
}

resource "google_compute_global_address" "sec_edgar_lb_ip" {
  count = local.enable_https_lb ? 1 : 0

  project    = var.project_id
  name       = "${var.cloud_run_service_name}-lb-ip"
  ip_version = "IPV4"

  depends_on = [google_project_service.required_apis]
}

resource "google_compute_managed_ssl_certificate" "sec_edgar_cert" {
  count = local.enable_https_lb ? 1 : 0

  project = var.project_id
  # Suffix lets the cert be replaced (create_before_destroy) when the domain changes.
  name = "${var.cloud_run_service_name}-cert-${substr(sha1(local.lb_domain), 0, 8)}"

  managed {
    domains = [local.lb_domain]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "google_compute_url_map" "sec_edgar_https" {
  count = local.enable_https_lb ? 1 : 0

  project         = var.project_id
  name            = "${var.cloud_run_service_name}-url-map"
  description     = "Routes all HTTPS traffic to the IAP-protected Cloud Run backend."
  default_service = google_compute_backend_service.sec_edgar_iap_backend[0].id
}

resource "google_compute_target_https_proxy" "sec_edgar_https" {
  count = local.enable_https_lb ? 1 : 0

  project          = var.project_id
  name             = "${var.cloud_run_service_name}-https-proxy"
  url_map          = google_compute_url_map.sec_edgar_https[0].id
  ssl_certificates = [google_compute_managed_ssl_certificate.sec_edgar_cert[0].id]
}

resource "google_compute_global_forwarding_rule" "sec_edgar_https" {
  count = local.enable_https_lb ? 1 : 0

  project               = var.project_id
  name                  = "${var.cloud_run_service_name}-https-fr"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_protocol           = "TCP"
  port_range            = "443"
  ip_address            = google_compute_global_address.sec_edgar_lb_ip[0].id
  target                = google_compute_target_https_proxy.sec_edgar_https[0].id
  labels                = var.labels
}

# --- HTTP (80) -> HTTPS (443) redirect ---------------------------------------
resource "google_compute_url_map" "sec_edgar_http_redirect" {
  count = local.enable_https_lb ? 1 : 0

  project     = var.project_id
  name        = "${var.cloud_run_service_name}-http-redirect"
  description = "Redirects plain HTTP to HTTPS."

  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "sec_edgar_http_redirect" {
  count = local.enable_https_lb ? 1 : 0

  project = var.project_id
  name    = "${var.cloud_run_service_name}-http-proxy"
  url_map = google_compute_url_map.sec_edgar_http_redirect[0].id
}

resource "google_compute_global_forwarding_rule" "sec_edgar_http_redirect" {
  count = local.enable_https_lb ? 1 : 0

  project               = var.project_id
  name                  = "${var.cloud_run_service_name}-http-fr"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_protocol           = "TCP"
  port_range            = "80"
  ip_address            = google_compute_global_address.sec_edgar_lb_ip[0].id
  target                = google_compute_target_http_proxy.sec_edgar_http_redirect[0].id
  labels                = var.labels
}
