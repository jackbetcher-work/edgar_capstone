view: v_financial_variances_yoy {
  sql_table_name: `@{GCP_PROJECT_ID}.@{BQ_DATASET_ID}.v_financial_variances_yoy` ;;
  drill_fields: [ticker, company_name, fiscal_year, total_revenue, avg_revenue_yoy_pct, avg_operating_income_yoy_pct, sec_filing_url]

  dimension: profile_id {
    primary_key: yes
    type: string
    sql: ${TABLE}.profile_id ;;
  }

  dimension: ticker {
    type: string
    label: "Ticker"
    sql: ${TABLE}.ticker ;;
  }

  dimension: company_name {
    type: string
    label: "Company Name"
    sql: ${TABLE}.company_name ;;
  }

  dimension: cik {
    type: string
    sql: ${TABLE}.cik ;;
  }

  dimension: gics_sector {
    type: string
    label: "GICS Sector"
    sql: ${TABLE}.gics_sector ;;
  }

  dimension: gics_sub_industry {
    type: string
    label: "GICS Sub-Industry"
    sql: ${TABLE}.gics_sub_industry ;;
  }

  dimension: fiscal_year {
    type: number
    label: "Fiscal Year"
    value_format_name: id
    sql: ${TABLE}.fiscal_year ;;
  }

  dimension: prev_fiscal_year {
    type: number
    label: "Prior Fiscal Year"
    value_format_name: id
    sql: ${TABLE}.prev_fiscal_year ;;
  }

  dimension: sec_filing_url {
    type: string
    label: "SEC Filing URL"
    sql: ${TABLE}.sec_filing_url ;;
    link: {
      label: "View SEC Form 10-K"
      url: "{{ value }}"
    }
  }

  dimension: unstructured_gcs_uri {
    type: string
    sql: ${TABLE}.unstructured_gcs_uri ;;
  }

  measure: total_revenue {
    type: sum
    label: "Revenue (USD)"
    value_format_name: usd_0
    sql: ${TABLE}.revenue ;;
  }

  measure: total_revenue_yoy_delta {
    type: sum
    label: "Revenue YoY Delta (USD)"
    value_format_name: usd_0
    sql: ${TABLE}.revenue_yoy_delta ;;
  }

  measure: avg_revenue_yoy_pct {
    type: average
    label: "Revenue YoY Growth (%)"
    value_format: "0.00\"%\""
    sql: ${TABLE}.revenue_yoy_pct ;;
  }

  measure: total_operating_income {
    type: sum
    label: "Operating Income (USD)"
    value_format_name: usd_0
    sql: ${TABLE}.operating_income ;;
  }

  measure: total_operating_income_yoy_delta {
    type: sum
    label: "Operating Income YoY Delta (USD)"
    value_format_name: usd_0
    sql: ${TABLE}.operating_income_yoy_delta ;;
  }

  measure: avg_operating_income_yoy_pct {
    type: average
    label: "Operating Income YoY Growth (%)"
    value_format: "0.00\"%\""
    sql: ${TABLE}.operating_income_yoy_pct ;;
  }

  measure: total_net_income {
    type: sum
    label: "Net Income (USD)"
    value_format_name: usd_0
    sql: ${TABLE}.net_income ;;
  }

  measure: total_net_income_yoy_delta {
    type: sum
    label: "Net Income YoY Delta (USD)"
    value_format_name: usd_0
    sql: ${TABLE}.net_income_yoy_delta ;;
  }

  measure: avg_net_income_yoy_pct {
    type: average
    label: "Net Income YoY Growth (%)"
    value_format: "0.00\"%\""
    sql: ${TABLE}.net_income_yoy_pct ;;
  }
}
