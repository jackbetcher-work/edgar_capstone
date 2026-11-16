- dashboard: sec_edgar_observability_and_cost
  title: "SEC EDGAR ADK Observability, Cost Savings & Security Audit"
  layout: newspaper
  preferred_viewer: dashboards-next
  description: "Real-time BI observability dashboard tracking Gemini 3.8 Flash & 3.5 Flash token spend, cached-token cost savings, TTFT/p95 latency SLO compliance, ADK tool usage, conversational session analytics, and 10-K access audit logs."
  refresh: 5 minutes

  filters:
    - name: Date_Range
      title: "Recorded Date Range"
      type: field_filter
      default_value: "7 days"
      allow_multiple_values: true
      required: false
      model: sec_edgar_analytics
      explore: llm_telemetry_and_cost
      field: llm_telemetry_events.recorded_date

    - name: Model_Tier
      title: "Gemini Model Tier"
      type: field_filter
      default_value: ""
      allow_multiple_values: true
      required: false
      model: sec_edgar_analytics
      explore: llm_telemetry_and_cost
      field: llm_telemetry_events.model_name

  elements:
    - name: kpi_total_llm_cost_and_cache_savings
      title: "Total LLM Spend & Context-Cache Discount Savings (USD)"
      model: sec_edgar_analytics
      explore: llm_telemetry_and_cost
      type: single_value
      fields:
        - llm_telemetry_events.total_cost_usd
        - llm_telemetry_events.cached_token_savings_usd
        - llm_telemetry_events.cache_cost_savings_pct
      listen:
        Date_Range: llm_telemetry_events.recorded_date
        Model_Tier: llm_telemetry_events.model_name
      row: 0
      col: 0
      width: 8
      height: 4

    - name: kpi_p95_ttft_slo_compliance
      title: "p95 TTFT (ms) & <=3.0s SLO Compliance Rate"
      model: sec_edgar_analytics
      explore: llm_telemetry_and_cost
      type: single_value
      fields:
        - llm_telemetry_events.p95_ttft_ms
        - llm_telemetry_events.ttft_slo_compliance_rate
      listen:
        Date_Range: llm_telemetry_events.recorded_date
        Model_Tier: llm_telemetry_events.model_name
      row: 0
      col: 8
      width: 8
      height: 4

    - name: kpi_sessions_and_grounded_reports
      title: "Analyst Sessions, Avg Session Length & Grounded Reports"
      model: sec_edgar_analytics
      explore: session_analytics
      type: single_value
      fields:
        - session_telemetry_events.distinct_sessions_count
        - session_telemetry_events.avg_session_length_turns
        - session_telemetry_events.reports_generated_count
      listen:
        Date_Range: session_telemetry_events.recorded_date
      row: 0
      col: 16
      width: 8
      height: 4

    - name: model_tier_cost_and_cached_token_savings
      title: "Token Spend vs Cached-Token Cost Savings by Model & ADK Agent"
      model: sec_edgar_analytics
      explore: llm_telemetry_and_cost
      type: looker_column
      fields:
        - llm_telemetry_events.recorded_date
        - llm_telemetry_events.model_name
        - llm_telemetry_events.agent_name
        - llm_telemetry_events.total_cost_usd
        - llm_telemetry_events.cached_token_savings_usd
        - llm_telemetry_events.cache_hit_rate
      sorts:
        - llm_telemetry_events.recorded_date desc
      listen:
        Date_Range: llm_telemetry_events.recorded_date
        Model_Tier: llm_telemetry_events.model_name
      row: 4
      col: 0
      width: 12
      height: 8

    - name: average_and_p95_latency_breakdown
      title: "Average & p95 TTFT / End-to-End Model Latency by Workload"
      model: sec_edgar_analytics
      explore: llm_telemetry_and_cost
      type: looker_line
      fields:
        - llm_telemetry_events.recorded_date
        - llm_telemetry_events.task_category
        - llm_telemetry_events.avg_ttft_ms
        - llm_telemetry_events.p95_ttft_ms
        - llm_telemetry_events.avg_total_latency_ms
        - llm_telemetry_events.p95_total_latency_ms
      sorts:
        - llm_telemetry_events.recorded_date asc
      listen:
        Date_Range: llm_telemetry_events.recorded_date
        Model_Tier: llm_telemetry_events.model_name
      row: 4
      col: 12
      width: 12
      height: 8

    - name: adk_tool_usage_and_intent_routing
      title: "ADK Tool Invocations, Companies Analyzed & Turn Latency by Intent"
      model: sec_edgar_analytics
      explore: session_analytics
      type: looker_grid
      fields:
        - session_telemetry_events.intent
        - session_telemetry_events.execution_mode
        - session_telemetry_events.turn_count
        - session_telemetry_events.total_companies_analyzed
        - session_telemetry_events.total_tool_invocations
        - session_telemetry_events.avg_turn_latency_ms
        - session_telemetry_events.p95_turn_latency_ms
        - session_telemetry_events.fact_check_pass_rate
        - session_telemetry_events.avg_evaluation_score
      sorts:
        - session_telemetry_events.turn_count desc
      listen:
        Date_Range: session_telemetry_events.recorded_date
      row: 12
      col: 0
      width: 12
      height: 8

    - name: sec_10k_access_and_security_audit_trail
      title: "10-K Report Access, IAP/Okta SSO & SPIFFE S2S Security Audit Trail"
      model: sec_edgar_analytics
      explore: security_audit_logs
      type: looker_grid
      fields:
        - security_audit_logs.recorded_time
        - security_audit_logs.event_type
        - security_audit_logs.status
        - security_audit_logs.user_email
        - security_audit_logs.identity_provider
        - security_audit_logs.agent_identity
        - security_audit_logs.resource
        - security_audit_logs.audit_event_count
      sorts:
        - security_audit_logs.recorded_time desc
      limit: 50
      listen:
        Date_Range: security_audit_logs.recorded_date
      row: 12
      col: 12
      width: 12
      height: 8
