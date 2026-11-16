view: llm_telemetry_events {
  sql_table_name: `@{GCP_PROJECT_ID}.@{BQ_DATASET_ID}.llm_telemetry_events` ;;
  drill_fields: [call_id, recorded_time, model_name, agent_name, task_category, total_cost_usd, cached_token_savings_usd, avg_ttft_ms]

  dimension: call_id {
    primary_key: yes
    type: string
    label: "LLM Call ID"
    sql: ${TABLE}.call_id ;;
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

  dimension: user_id {
    type: string
    label: "Analyst Principal"
    sql: ${TABLE}.user_id ;;
  }

  dimension: agent_name {
    type: string
    label: "ADK Agent"
    sql: ${TABLE}.agent_name ;;
  }

  dimension: model_name {
    type: string
    label: "Gemini Model Tier"
    sql: ${TABLE}.model_name ;;
  }

  dimension: task_category {
    type: string
    label: "Workload Task Category"
    sql: ${TABLE}.task_category ;;
  }

  dimension: react_phase {
    type: string
    label: "ReAct Loop Phase"
    sql: ${TABLE}.react_phase ;;
  }

  dimension: cache_hit {
    type: yesno
    label: "Context Cache Hit"
    sql: ${TABLE}.cache_hit ;;
  }

  dimension: within_ttft_slo {
    type: yesno
    label: "Within 3.0s TTFT SLO"
    sql: ${TABLE}.within_ttft_slo ;;
  }

  measure: call_count {
    type: count
    label: "LLM Calls"
  }

  measure: total_context_tokens {
    type: sum
    label: "Total Input Context Tokens"
    sql: ${TABLE}.total_context_tokens ;;
  }

  measure: cached_context_tokens {
    type: sum
    label: "Cached Context Tokens"
    sql: ${TABLE}.cached_context_tokens ;;
  }

  measure: uncached_context_tokens {
    type: sum
    label: "Uncached Context Tokens"
    sql: ${TABLE}.uncached_context_tokens ;;
  }

  measure: total_output_tokens {
    type: sum
    label: "Generated Output + Reasoning Tokens"
    sql: ${TABLE}.output_tokens + COALESCE(${TABLE}.thoughts_tokens, 0) ;;
  }

  measure: cache_hit_rate {
    type: number
    label: "Context Cache Hit Rate"
    value_format_name: percent_1
    sql: SAFE_DIVIDE(COUNTIF(${TABLE}.cache_hit), NULLIF(COUNT(*), 0)) ;;
  }

  measure: avg_total_latency_ms {
    type: average
    label: "Average Model Latency (ms)"
    value_format: "0.0"
    sql: ${TABLE}.total_model_latency_ms ;;
  }

  measure: p95_total_latency_ms {
    type: percentile
    percentile: 95
    label: "p95 Model Latency (ms)"
    value_format: "0.0"
    sql: ${TABLE}.total_model_latency_ms ;;
  }

  measure: avg_context_latency_ms {
    type: average
    label: "Average Context Prefill Latency (ms)"
    value_format: "0.0"
    sql: ${TABLE}.context_latency_ms ;;
  }

  measure: p95_context_latency_ms {
    type: percentile
    percentile: 95
    label: "p95 Context Prefill Latency (ms)"
    value_format: "0.0"
    sql: ${TABLE}.context_latency_ms ;;
  }

  measure: avg_ttft_ms {
    type: average
    label: "Average TTFT (ms)"
    value_format: "0.0"
    sql: ${TABLE}.ttft_ms ;;
  }

  measure: p95_ttft_ms {
    type: percentile
    percentile: 95
    label: "p95 TTFT (ms)"
    value_format: "0.0"
    sql: ${TABLE}.ttft_ms ;;
  }

  measure: ttft_slo_compliance_rate {
    type: number
    label: "TTFT SLO Compliance Rate (<=3000ms)"
    value_format_name: percent_2
    sql: SAFE_DIVIDE(COUNTIF(${TABLE}.within_ttft_slo), NULLIF(COUNT(*), 0)) ;;
  }

  measure: estimated_cache_latency_saved_ms {
    type: sum
    label: "Estimated Prefill Latency Saved via Cache (ms)"
    value_format: "0.0"
    sql: ${TABLE}.estimated_cache_latency_saved_ms ;;
  }

  measure: total_cost_usd {
    type: sum
    label: "Total LLM Spend (USD)"
    value_format: "$0.0000"
    sql: ${TABLE}.total_cost_usd ;;
  }

  measure: uncached_baseline_cost_usd {
    type: sum
    label: "Uncached Baseline Spend (USD)"
    value_format: "$0.0000"
    sql: ${TABLE}.uncached_baseline_cost_usd ;;
  }

  measure: cached_token_savings_usd {
    type: sum
    label: "Cached-Token Cost Savings (USD)"
    value_format: "$0.0000"
    sql: ${TABLE}.cached_token_savings_usd ;;
  }

  measure: cache_cost_savings_pct {
    type: number
    label: "Cached-Token Cost Discount (%)"
    value_format_name: percent_1
    sql: SAFE_DIVIDE(${cached_token_savings_usd}, NULLIF(${uncached_baseline_cost_usd}, 0)) ;;
  }
}
