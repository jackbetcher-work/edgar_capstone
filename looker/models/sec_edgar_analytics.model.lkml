connection: "@{BIGQUERY_CONNECTION}"

include: "/views/*.view.lkml"
include: "/dashboards/*.dashboard.lookml"

datagroup: sec_edgar_filings_datagroup {
  label: "SEC EDGAR 10-K Filings Refresh"
  description: "Refreshes Looker explores when new SEC Form 10-K financial profiles are ingested into BigQuery."
  sql_trigger: SELECT MAX(filing_date) FROM `@{GCP_PROJECT_ID}.@{BQ_DATASET_ID}.financial_metrics` ;;
  max_cache_age: "6 hours"
}

datagroup: sec_edgar_streaming_telemetry_datagroup {
  label: "SEC EDGAR Streaming Telemetry & Audit Refresh"
  description: "Near-real-time cache policy for streaming LLM cost/latency telemetry and 10-K security audit sinks."
  sql_trigger: SELECT TIMESTAMP_TRUNC(MAX(recorded_at), MINUTE) FROM `@{GCP_PROJECT_ID}.@{BQ_DATASET_ID}.llm_telemetry_events` ;;
  max_cache_age: "5 minutes"
}

persist_with: sec_edgar_filings_datagroup

explore: yoy_variances {
  label: "SEC 10-K YoY Financial Variances"
  description: "Deterministic year-over-year Revenue, Operating Income, and Net Income variances backed by v_financial_variances_yoy."
  view_name: v_financial_variances_yoy

  join: financial_metrics {
    type: left_outer
    relationship: one_to_one
    sql_on: ${v_financial_variances_yoy.profile_id} = ${financial_metrics.profile_id} ;;
  }
}

explore: sector_benchmarks {
  label: "S&P 500 GICS Sector Benchmarks"
  description: "Cross-issuer GICS sector revenue, operating income, net income, and margin benchmarks."
  from: financial_metrics
  view_name: sector_benchmarks
}

explore: peer_comparison {
  label: "Multi-Company Peer Comparison"
  description: "Side-by-side peer financial performance, YoY growth rates, and margin basis-point comparisons."
  from: v_financial_variances_yoy
  view_name: peer_comparison
}

explore: chunk_health {
  label: "RAG Vector DB Chunk Health & Coverage"
  description: "768-dimensional embedding coverage, section completeness (Item 1A & Item 7), and token distribution diagnostics."
  view_name: v_chunk_health_metrics
}

explore: llm_telemetry_and_cost {
  label: "LLM Token Cost, Cache Savings & Latency"
  description: "Streaming Gemini 2.5 Pro & Gemini 3.5 Flash token consumption, cached-token USD savings, TTFT SLO compliance, and prefill/decode latency."
  persist_with: sec_edgar_streaming_telemetry_datagroup
  view_name: llm_telemetry_events
}

explore: session_analytics {
  label: "Conversational Sessions, Tool Usage & Reports"
  description: "Session length, companies analyzed, grounded report generation counts, ADK tool invocations, and Flash judge rubric scores."
  persist_with: sec_edgar_streaming_telemetry_datagroup
  view_name: session_telemetry_events

  join: llm_telemetry_events {
    type: left_outer
    relationship: one_to_many
    sql_on: ${session_telemetry_events.session_id} = ${llm_telemetry_events.session_id} ;;
  }
}

explore: security_audit_logs {
  label: "10-K Access & IAP/S2S Security Audit Trail"
  description: "Immutable IAP/Workspace/Okta authentication logs, per-agent SPIFFE S2S attestations, and 10-K filing access audit records."
  persist_with: sec_edgar_streaming_telemetry_datagroup
  view_name: security_audit_logs
}
