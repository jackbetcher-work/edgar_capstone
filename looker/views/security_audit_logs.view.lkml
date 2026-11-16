view: security_audit_logs {
  sql_table_name: `@{GCP_PROJECT_ID}.@{BQ_DATASET_ID}.security_audit_logs` ;;
  drill_fields: [event_id, recorded_time, event_type, status, user_email, identity_provider, resource, agent_identity]

  dimension: event_id {
    primary_key: yes
    type: string
    label: "Audit Event ID"
    sql: ${TABLE}.event_id ;;
  }

  dimension_group: recorded {
    type: time
    label: "Recorded"
    timeframes: [raw, time, hour, date, week, month]
    sql: ${TABLE}.recorded_at ;;
  }

  dimension: event_type {
    type: string
    label: "Audit Event Type"
    sql: ${TABLE}.event_type ;;
  }

  dimension: status {
    type: string
    label: "Authorization Outcome"
    sql: ${TABLE}.status ;;
  }

  dimension: resource {
    type: string
    label: "API Resource Path"
    sql: ${TABLE}.resource ;;
  }

  dimension: user_email {
    type: string
    label: "Principal Email"
    sql: ${TABLE}.user_email ;;
  }

  dimension: hosted_domain {
    type: string
    label: "Workspace Hosted Domain"
    sql: ${TABLE}.hosted_domain ;;
  }

  dimension: identity_provider {
    type: string
    label: "Identity Provider"
    sql: ${TABLE}.identity_provider ;;
  }

  dimension: agent_identity {
    type: string
    label: "ADK S2S Agent Identity"
    sql: ${TABLE}.agent_identity ;;
  }

  dimension: spiffe_id {
    type: string
    label: "SPIFFE Workload ID"
    sql: ${TABLE}.spiffe_id ;;
  }

  dimension: iap_assertion_verified {
    type: yesno
    label: "IAP Assertion Verified"
    sql: ${TABLE}.iap_assertion_verified ;;
  }

  dimension: detail {
    type: string
    label: "Audit Detail"
    sql: ${TABLE}.detail ;;
  }

  measure: audit_event_count {
    type: count
    label: "Total Audit Events"
  }

  measure: denied_event_count {
    type: count
    label: "Denied Access Attempts (401/403/Blocked)"
    filters: [status: "denied_401, denied_403, blocked"]
  }

  measure: filing_10k_access_count {
    type: count
    label: "10-K Report & Section Access Events"
    filters: [event_type: "conversational_rag_chat, filing_10k_section_read, rag_chunk_search, variance_calculation"]
  }

  measure: distinct_authenticated_users {
    type: count_distinct
    label: "Distinct Principals"
    sql: ${user_email} ;;
  }

  measure: s2s_agent_attested_count {
    type: number
    label: "S2S SPIFFE Attested Events"
    sql: COUNTIF(${TABLE}.spiffe_id IS NOT NULL AND ${TABLE}.spiffe_id != '') ;;
  }

  dimension: trace_id {
    type: string
    label: "Cloud Trace ID"
    description: "OpenTelemetry trace ID of the originating API request. Links to the end-to-end latency trace in Cloud Trace."
    sql: ${TABLE}.trace_id ;;
    link: {
      label: "Open in Cloud Trace"
      url: "https://console.cloud.google.com/traces/list?project=@{GCP_PROJECT_ID}&tid={{ value }}"
    }
  }
}
