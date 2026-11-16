project_name: "sec_edgar_analytics"

constant: GCP_PROJECT_ID {
  value: "iron-radio-514217-q7"
  export: override_optional
}

constant: BQ_DATASET_ID {
  value: "sec_edgar"
  export: override_optional
}

constant: BIGQUERY_CONNECTION {
  value: "sec_edgar_bigquery"
  export: override_optional
}
