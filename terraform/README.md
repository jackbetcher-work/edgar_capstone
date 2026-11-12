# Terraform — SEC EDGAR BigQuery & Metadata-Filtered RAG Foundation

Provisions the BigQuery data & RAG foundation in project `iron-radio-514217-q7` (`us-central1`) with remote Terraform state stored in `gs://iron-radio-514217-q7-tfstate`.

## Provisioned Resources

- **Remote State Bucket** (`gs://iron-radio-514217-q7-tfstate`): Bootstrapped with object versioning, uniform bucket-level access, and enforced public access prevention.
- **BigQuery Dataset (`sec_edgar`, `us-central1`):**
  - `financial_metrics`: Native table partitioned by `fiscal_year` (`2015–2035`) and clustered by `gics_sector, ticker`, automatically seeded from `gs://fde-sec-edgar-sandbox-dev-sec-corpus/manifests/bigquery_financials.jsonl`.
  - `financial_metrics_ext`: Live GCS external table over `manifests/bigquery_financials.jsonl`.
  - `filing_chunks`: Metadata-filtered RAG table partitioned by `fiscal_year` and clustered by `gics_sector, ticker, section_id` with `embedding` (`ARRAY<FLOAT64>`) ready for `VECTOR_SEARCH`.
  - `unstructured_10k_objects`: BigQuery Object Table indexing `gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/*`.
  - `v_financial_variances_yoy`: Deterministic YoY variance SQL view (`revenue`, `operating_income`, `net_income` deltas and percentages).
- **BigQuery Cloud Resource Connection (`vertex-ai-rag-conn`):** Granted `roles/aiplatform.user` and `roles/storage.objectViewer` on the SEC corpus bucket.
- **Runtime Service Account (`sec-analyst-sa@iron-radio-514217-q7.iam.gserviceaccount.com`):** Granted least-privilege BigQuery job/connection/dataset permissions and read-only GCS corpus access.

## Usage

```bash
# 1. Bootstrap the versioned GCS remote state bucket (idempotent)
../.venv/bin/python ../scripts/bootstrap_tfstate_bucket.py

# 2. Initialize Terraform with the GCS backend
cd terraform
terraform init

# 3. Validate, plan, and apply
terraform validate
terraform plan
terraform apply
```

## Example Metadata-Filtered RAG Query (`VECTOR_SEARCH`)

Once chunks and embeddings are populated into `sec_edgar.filing_chunks`, pre-filtered vector search runs natively in SQL:

```sql
SELECT
  base.ticker,
  base.company_name,
  base.fiscal_year,
  base.section_id,
  base.chunk_text,
  base.sec_filing_url,
  distance
FROM VECTOR_SEARCH(
  (
    SELECT *
    FROM `iron-radio-514217-q7.sec_edgar.filing_chunks`
    WHERE gics_sector = 'Information Technology'
      AND fiscal_year BETWEEN 2024 AND 2026
      AND section_id = 'item_1a'
  ),
  'embedding',
  (SELECT @query_embedding AS embedding),
  top_k => 10,
  distance_type => 'COSINE'
);
```
