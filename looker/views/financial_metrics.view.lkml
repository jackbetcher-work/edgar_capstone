view: financial_metrics {
  sql_table_name: `@{GCP_PROJECT_ID}.@{BQ_DATASET_ID}.financial_metrics` ;;
  drill_fields: [profile_id, ticker, company_name, fiscal_year, total_revenue, total_operating_income, total_net_income]

  dimension: profile_id {
    primary_key: yes
    type: string
    description: "Unique company-year profile identifier ({TICKER}_{FY})"
    sql: ${TABLE}.profile_id ;;
  }

  dimension: ticker {
    type: string
    label: "Ticker Symbol"
    sql: ${TABLE}.ticker ;;
  }

  dimension: cik {
    type: string
    label: "SEC CIK"
    sql: ${TABLE}.cik ;;
  }

  dimension: company_name {
    type: string
    label: "Company Name"
    sql: ${TABLE}.company_name ;;
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

  dimension_group: filing {
    type: time
    timeframes: [raw, date, month, quarter, year]
    convert_tz: no
    datatype: date
    sql: ${TABLE}.filing_date ;;
  }

  dimension_group: period_of_report {
    type: time
    timeframes: [raw, date, quarter, year]
    convert_tz: no
    datatype: date
    sql: ${TABLE}.period_of_report ;;
  }

  dimension: accession_number {
    type: string
    sql: ${TABLE}.accession_number ;;
  }

  dimension: sec_filing_url {
    type: string
    label: "SEC EDGAR 10-K URL"
    sql: ${TABLE}.sec_filing_url ;;
    link: {
      label: "Open Official SEC Form 10-K"
      url: "{{ value }}"
    }
  }

  dimension: operating_margin_pct {
    type: number
    label: "Operating Margin (%)"
    value_format: "0.00\"%\""
    sql: SAFE_DIVIDE(${TABLE}.operating_income, ABS(${TABLE}.revenue)) * 100.0 ;;
  }

  dimension: net_margin_pct {
    type: number
    label: "Net Margin (%)"
    value_format: "0.00\"%\""
    sql: SAFE_DIVIDE(${TABLE}.net_income, ABS(${TABLE}.revenue)) * 100.0 ;;
  }

  measure: company_count {
    type: count_distinct
    label: "Distinct Issuers"
    sql: ${ticker} ;;
  }

  measure: filing_count {
    type: count
    label: "10-K Filings Count"
  }

  measure: total_revenue {
    type: sum
    label: "Total Revenue (USD)"
    value_format_name: usd_0
    sql: ${TABLE}.revenue ;;
  }

  measure: average_company_revenue {
    type: average
    label: "Average Issuer Revenue (USD)"
    value_format_name: usd_0
    sql: ${TABLE}.revenue ;;
  }

  measure: total_operating_income {
    type: sum
    label: "Total Operating Income (USD)"
    value_format_name: usd_0
    sql: ${TABLE}.operating_income ;;
  }

  measure: total_net_income {
    type: sum
    label: "Total Net Income (USD)"
    value_format_name: usd_0
    sql: ${TABLE}.net_income ;;
  }

  measure: average_operating_margin_pct {
    type: average
    label: "Average Operating Margin (%)"
    value_format: "0.00\"%\""
    sql: ${operating_margin_pct} ;;
  }

  measure: average_net_margin_pct {
    type: average
    label: "Average Net Margin (%)"
    value_format: "0.00\"%\""
    sql: ${net_margin_pct} ;;
  }
}
