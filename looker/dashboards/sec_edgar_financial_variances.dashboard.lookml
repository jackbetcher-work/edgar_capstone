- dashboard: sec_edgar_financial_variances
  title: "SEC EDGAR 10-K Longitudinal Variances, Sector Benchmarks & Chunk Health"
  layout: newspaper
  preferred_viewer: dashboards-next
  description: "Executive financial analysis & RAG corpus health dashboard powered by deterministic BigQuery views (v_financial_variances_yoy, financial_metrics, v_chunk_health_metrics)."
  refresh: 1 hour

  filters:
    - name: GICS_Sector
      title: "GICS Sector"
      type: field_filter
      default_value: ""
      allow_multiple_values: true
      required: false
      model: sec_edgar_analytics
      explore: yoy_variances
      field: v_financial_variances_yoy.gics_sector

    - name: Ticker_Filter
      title: "Ticker Symbol"
      type: field_filter
      default_value: "NVDA,AAPL,MSFT,GOOGL"
      allow_multiple_values: true
      required: false
      model: sec_edgar_analytics
      explore: yoy_variances
      field: v_financial_variances_yoy.ticker

  elements:
    - name: multi_company_yoy_revenue_and_operating_variances
      title: "Multi-Company YoY Revenue & Operating Income Variances"
      model: sec_edgar_analytics
      explore: yoy_variances
      type: looker_grid
      fields:
        - v_financial_variances_yoy.ticker
        - v_financial_variances_yoy.company_name
        - v_financial_variances_yoy.gics_sector
        - v_financial_variances_yoy.fiscal_year
        - v_financial_variances_yoy.total_revenue
        - v_financial_variances_yoy.avg_revenue_yoy_pct
        - v_financial_variances_yoy.total_operating_income
        - v_financial_variances_yoy.avg_operating_income_yoy_pct
        - v_financial_variances_yoy.total_net_income
        - v_financial_variances_yoy.avg_net_income_yoy_pct
        - v_financial_variances_yoy.sec_filing_url
      sorts:
        - v_financial_variances_yoy.fiscal_year desc
        - v_financial_variances_yoy.total_revenue desc
      listen:
        GICS_Sector: v_financial_variances_yoy.gics_sector
        Ticker_Filter: v_financial_variances_yoy.ticker
      row: 0
      col: 0
      width: 24
      height: 8

    - name: gics_sector_revenue_and_margin_benchmarks
      title: "S&P 500 GICS Sector Revenue & Operating Margin Benchmarks"
      model: sec_edgar_analytics
      explore: sector_benchmarks
      type: looker_column
      fields:
        - sector_benchmarks.gics_sector
        - sector_benchmarks.fiscal_year
        - sector_benchmarks.company_count
        - sector_benchmarks.total_revenue
        - sector_benchmarks.average_operating_margin_pct
        - sector_benchmarks.average_net_margin_pct
      sorts:
        - sector_benchmarks.fiscal_year desc
        - sector_benchmarks.total_revenue desc
      row: 8
      col: 0
      width: 12
      height: 8

    - name: rag_vector_db_chunk_health_scorecard
      title: "RAG Vector DB 768-d Embedding Coverage & Section Health"
      model: sec_edgar_analytics
      explore: chunk_health
      type: looker_grid
      fields:
        - v_chunk_health_metrics.gics_sector
        - v_chunk_health_metrics.section_id
        - v_chunk_health_metrics.health_status
        - v_chunk_health_metrics.avg_health_score
        - v_chunk_health_metrics.total_chunks_sum
        - v_chunk_health_metrics.avg_embedding_coverage
        - v_chunk_health_metrics.avg_filing_coverage
        - v_chunk_health_metrics.p95_chunk_tokens
        - v_chunk_health_metrics.anomaly_chunk_count
      sorts:
        - v_chunk_health_metrics.total_chunks_sum desc
      row: 8
      col: 12
      width: 12
      height: 8

    - name: multi_company_peer_margin_comparison
      title: "Multi-Company Peer Revenue & Operating Margin Comparison"
      model: sec_edgar_analytics
      explore: peer_comparison
      type: looker_bar
      fields:
        - financial_metrics.ticker
        - financial_metrics.company_name
        - financial_metrics.gics_sector
        - financial_metrics.fiscal_year
        - financial_metrics.total_revenue
        - financial_metrics.average_operating_margin_pct
        - financial_metrics.average_net_margin_pct
      sorts:
        - financial_metrics.fiscal_year desc
        - financial_metrics.total_revenue desc
      row: 16
      col: 0
      width: 24
      height: 8
