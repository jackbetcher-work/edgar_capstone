view: session_telemetry_events {
  sql_table_name: `@{GCP_PROJECT_ID}.@{BQ_DATASET_ID}.session_telemetry_events` ;;
  drill_fields: [turn_id, recorded_time, session_id, user_id, intent, execution_mode, reports_generated_count, avg_turn_latency_ms]

  dimension: turn_id {
    primary_key: yes
    type: string
    label: "Turn Telemetry ID"
    sql: ${TABLE}.turn_id ;;
  }

  dimension_group: recorded {
    type: time
    label: "Recorded"
    timeframes: [raw, time, hour, date, week, month]
    sql: ${TABLE}.recorded_at ;;
  }

  dimension: session_id {
    type: string
    label: "Session ID"
    sql: ${TABLE}.session_id ;;
  }

  dimension: turn_index {
    type: number
    label: "Turn Index"
    sql: ${TABLE}.turn_index ;;
  }

  dimension: user_id {
    type: string
    label: "Analyst Email"
    sql: ${TABLE}.user_id ;;
  }

  dimension: intent {
    type: string
    label: "Routing Intent"
    sql: ${TABLE}.intent ;;
  }

  dimension: execution_mode {
    type: string
    label: "Execution Mode"
    sql: ${TABLE}.execution_mode ;;
  }

  dimension: status {
    type: string
    label: "Turn Status"
    sql: ${TABLE}.status ;;
  }

  dimension: gics_sector {
    type: string
    label: "GICS Sector Filter"
    sql: ${TABLE}.gics_sector ;;
  }

  dimension: report_generated {
    type: yesno
    label: "Grounded Report Generated"
    sql: ${TABLE}.report_generated ;;
  }

  dimension: fact_check_passed {
    type: yesno
    label: "Numeric Fact-Check Passed"
    sql: ${TABLE}.fact_check_passed ;;
  }

  dimension: evaluation_grade {
    type: string
    label: "Flash Judge Grade"
    sql: ${TABLE}.evaluation_grade ;;
  }

  measure: turn_count {
    type: count
    label: "Conversational Turns"
  }

  measure: distinct_sessions_count {
    type: count_distinct
    label: "Distinct Analyst Sessions"
    sql: ${session_id} ;;
  }

  measure: avg_session_length_turns {
    type: number
    label: "Average Session Length (Turns)"
    value_format: "0.00"
    sql: SAFE_DIVIDE(COUNT(*), NULLIF(COUNT(DISTINCT ${session_id}), 0)) ;;
  }

  measure: max_session_elapsed_seconds {
    type: max
    label: "Max Session Duration (s)"
    value_format: "0.0"
    sql: ${TABLE}.session_elapsed_seconds ;;
  }

  measure: reports_generated_count {
    type: count
    label: "Grounded 10-K Reports Generated"
    filters: [report_generated: "yes"]
  }

  measure: total_companies_analyzed {
    type: sum
    label: "Company Scopes Analyzed"
    sql: ${TABLE}.companies_analyzed_count ;;
  }

  measure: total_tool_invocations {
    type: sum
    label: "Total ADK Tool Invocations"
    sql: ${TABLE}.tool_invocation_count ;;
  }

  measure: avg_turn_latency_ms {
    type: average
    label: "Average End-to-End Turn Latency (ms)"
    value_format: "0.0"
    sql: ${TABLE}.total_latency_ms ;;
  }

  measure: p95_turn_latency_ms {
    type: percentile
    percentile: 95
    label: "p95 End-to-End Turn Latency (ms)"
    value_format: "0.0"
    sql: ${TABLE}.total_latency_ms ;;
  }

  measure: avg_evaluation_score {
    type: average
    label: "Average Flash Judge Rubric Score"
    value_format: "0.000"
    sql: ${TABLE}.evaluation_score ;;
  }

  measure: fact_check_pass_rate {
    type: number
    label: "Deterministic Fact-Check Pass Rate"
    value_format_name: percent_2
    sql: SAFE_DIVIDE(COUNTIF(${TABLE}.fact_check_passed), NULLIF(COUNT(*), 0)) ;;
  }

  measure: total_session_cost_usd {
    type: sum
    label: "Session LLM Spend (USD)"
    value_format: "$0.0000"
    sql: ${TABLE}.total_cost_usd ;;
  }

  measure: total_session_cache_savings_usd {
    type: sum
    label: "Session Cached-Token Savings (USD)"
    value_format: "$0.0000"
    sql: ${TABLE}.cached_token_savings_usd ;;
  }
}
