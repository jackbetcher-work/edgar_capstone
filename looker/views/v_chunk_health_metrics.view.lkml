view: v_chunk_health_metrics {
  sql_table_name: `@{GCP_PROJECT_ID}.@{BQ_DATASET_ID}.v_chunk_health_metrics` ;;
  drill_fields: [gics_sector, section_id, health_status, avg_health_score, total_chunks_sum, avg_embedding_coverage]

  dimension: rollup_key {
    primary_key: yes
    hidden: yes
    type: string
    sql: CONCAT(${TABLE}.gics_sector, '::', ${TABLE}.section_id) ;;
  }

  dimension: gics_sector {
    type: string
    label: "GICS Sector Rollup"
    sql: ${TABLE}.gics_sector ;;
  }

  dimension: section_id {
    type: string
    label: "10-K Section ID"
    sql: ${TABLE}.section_id ;;
  }

  dimension: health_status {
    type: string
    label: "Health Status"
    sql: ${TABLE}.health_status ;;
  }

  measure: total_chunks_sum {
    type: sum
    label: "Total Indexed Chunks"
    sql: ${TABLE}.total_chunks ;;
  }

  measure: distinct_profiles_sum {
    type: sum
    label: "Indexed Company-Year Profiles"
    sql: ${TABLE}.distinct_profiles ;;
  }

  measure: avg_embedding_coverage {
    type: average
    label: "768-d Embedding Coverage Ratio"
    value_format_name: percent_2
    sql: ${TABLE}.embedding_coverage_ratio ;;
  }

  measure: avg_filing_coverage {
    type: average
    label: "Filing Section Completeness Ratio"
    value_format_name: percent_2
    sql: ${TABLE}.filing_coverage_ratio ;;
  }

  measure: avg_health_score {
    type: average
    label: "Composite Health Score (0-100)"
    value_format: "0.00"
    sql: ${TABLE}.health_score ;;
  }

  measure: p50_chunk_tokens {
    type: average
    label: "p50 Chunk Token Estimate"
    value_format: "0"
    sql: ${TABLE}.p50_token_estimate ;;
  }

  measure: p95_chunk_tokens {
    type: max
    label: "p95 Chunk Token Estimate"
    value_format: "0"
    sql: ${TABLE}.p95_token_estimate ;;
  }

  measure: anomaly_chunk_count {
    type: sum
    label: "Anomaly Chunks (Undersized/Oversized/Missing Meta)"
    sql: ${TABLE}.undersized_chunk_count + ${TABLE}.oversized_chunk_count + ${TABLE}.missing_metadata_count ;;
  }
}
