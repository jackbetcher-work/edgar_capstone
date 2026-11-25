# Architecture Decision Records (ADRs)

This document captures the key architectural decisions, alternatives considered, and trade-offs behind the **SEC EDGAR Natural Language Analyst** platform.

---

## ADR-001: Hybrid RAG via BigQuery `VECTOR_SEARCH` + Vertex AI Search (Discovery Engine)

- **Status:** Accepted
- **Context:** Analysts ask both exact lexical/entity queries (e.g., specific product names, litigation terms like "PFAS" or "July 19 outage", GAAP line items) and broad semantic/thematic questions across ~501 S&P 500 companies (FY2020–FY2026 Item 1A and Item 7 filings).
- **Decision:** Run **BigQuery `VECTOR_SEARCH`** over 768-d `text-embedding-005` embeddings in `sec_edgar.filing_chunks` (partitioned by `fiscal_year`, clustered by `gics_sector, ticker, section_id`) **in parallel** with **Vertex AI Search (Discovery Engine, Enterprise tier + LLM add-on)**, fusing candidates with Reciprocal Rank Fusion (RRF), lexical overlap boosting, and boundary-first per-company-year coverage slicing (`hybrid_rag.py`). For sector-wide sweeps (~76 IT issuers), run a single BigQuery `ML.DISTANCE(..., 'COSINE')` + `QUALIFY ROW_NUMBER()` query (`vector_search_sector_top_per_ticker`).
- **Alternatives Considered:**
  1. *Vertex AI Vector Search (Matching Engine):* Higher always-on endpoint cost for a sandbox/curated S&P 500 corpus and requires maintaining a separate metadata sync pipeline alongside BigQuery.
  2. *BigQuery vector search alone:* Lower recall on exact acronyms, rare litigation terms, and extractive segment highlighting compared to Discovery Engine.
  3. *Vertex AI Search alone:* Does not support exact per-company-year top-K ranking across an entire GICS sector in a single query.
- **Consequences:**
  - *Pros:* Serverless scale-to-zero cost profile, strict SQL metadata pre-filtering (`ticker`, `fiscal_year`, `gics_sector`, `section_id`), fault-tolerant hybrid retrieval (if either branch fails with a non-retryable error, the other still serves), and single-RPC sector-wide scans in ~2–3 s.
  - *Cons / Production Evolution:* BigQuery query concurrency and slot limits constrain high-QPS multi-tenant scale; at hundreds of QPS, live lookups should migrate to a low-latency operational store and an autoscaling vector tier (see production scaling roadmap).

---

## ADR-002: Deterministic XBRL Variance Engine & Fail-Closed Claim Verification

- **Status:** Accepted
- **Context:** In financial research, arithmetic hallucinations, unit scaling mistakes (millions vs. billions in 10-K Markdown tables), or ungrounded citations invalidate analyst trust and create regulatory/audit exposure.
- **Decision:** Enforce **zero LLM arithmetic** and **fail-closed claim verification**:
  1. All numerical deltas, YoY %, CAGRs, operating/net margin bps shifts, R&D intensity, and 5 anomaly rules (`flag_material_variances`) are computed deterministically in Python (`variance_engine.py`, `material_variances.py`) from XBRL `financial_metrics` with full SEC accession numbers and `us-gaap` concepts.
  2. `FinancialAnalystAgent` outputs constrained JSON (`AnalystReport` via ADK `output_schema`), selecting `key_metrics` references (`ticker` + `metric`) rather than authoring metric tables.
  3. `structured_report.py` (`verify_analyst_report`) checks every finding's `chunk_id` resolution, verbatim `evidence_quote` presence in the cited chunk/section, and table-aware numeric fidelity (`_extract_table_aware_quote_figures`). If verification fails or drops an in-scope peer/boundary year, a single tool-less, low-thinking repair pass runs; if still unverified, the answer is **withheld** (`answer_verified: false`) rather than falling back to unverified prose.
- **Alternatives Considered:**
  1. *Free-form Markdown generation with post-hoc LLM judge only:* Higher risk of subtle numeric drift, fabricated `chunk_id`s, or unit mismatches that an LLM judge can miss.
  2. *Template-only narrative fallback on verification failure:* Masks synthesis errors and degrades analyst trust.
- **Consequences:**
  - *Pros:* Deterministic 100% numeric provenance, clickable inline XBRL & 10-K source chips, and zero unverified claims released to the UI.
  - *Cons:* Strict verbatim quote checking can occasionally reject valid paraphrased findings, triggering a repair pass (+1 LLM round-trip) or withholding an answer when quotes diverge from source formatting.

---

## ADR-003: Three-Tier Gemini Model Routing & Prompt Digest Compaction

- **Status:** Accepted
- **Context:** Financial reasoning requires strong multi-step tool orchestration and synthesis, while sector map-reduce, scope continuity, event compaction, and online evaluation must stay fast and cost-efficient under a 3.0 s p95 TTFT SLO (`DEFAULT_TTFT_SLO_MS = 3000.0`).
- **Decision:** Route workloads across three model/thinking tiers in `model_config.py`:
  - **`gemini-3.8-flash` (native thinking, `EDGAR_ANALYST_MODEL`):** `FinancialAnalystAgent` primary reasoning and ReAct tool orchestration.
  - **`gemini-3.8-flash` (`ThinkingLevel.LOW`, `EDGAR_FLASH_MODEL`):** `NarrativeEvaluatorAgent` LLM-as-a-Judge and verification repair passes. Empirically A/B tested over the golden suite where native `ThinkingLevel.LOW` outperformed legacy `thinking_budget=512` on both TTFT and citation coverage.
  - **`gemini-3.5-flash-lite` (`EDGAR_LITE_MODEL`):** High-throughput parallel `SectorMapSummarizerAgent` batches (~14 issuers/batch), `ScopeContinuityAgent` (`ThinkingLevel.MINIMAL`), and ADK `LlmEventSummarizer` compaction.
  - Pair model tiering with ADK `ContextCacheConfig` (min 2,048 tokens, 30 min TTL), a persistent background asyncio loop (`_PersistentAdkLoopManager`) to preserve HTTP/2 keepalive pools, and `>10×` variance payload compaction (`compact_variance_reports_for_prompt`) keeping full XBRL provenance in ADK `session_state`.
- **Alternatives Considered:**
  1. *Single flagship reasoning model (`gemini-2.5-pro`) for all tasks:* Higher token cost and TTFT latency; also unavailable in the target sandbox project.
  2. *Passing full ~18 KB/company XBRL JSON payloads in prompts:* Inflates prefill tokens and TTFT without improving narrative quality.
- **Consequences:**
  - *Pros:* Balanced quality, latency, and cost; all tiers configurable via environment variables without code changes.
  - *Cons:* Gemini models are served on the `global` Vertex AI endpoint (`EDGAR_GEMINI_LOCATION=global`) while data services run in `us-central1`, requiring explicit dual-location configuration.

---

## ADR-004: Direct Cloud Run IAP & Least-Privilege Custom SA (and Production Network Path)

- **Status:** Accepted (Dev/Capstone) — Production Network Upgrade Defined
- **Context:** The application must never be publicly anonymous, must enforce corporate identity/RBAC, and must operate under least-privilege IAM without default Compute Engine service accounts or static secrets.
- **Decision:**
  1. Enable **Direct Cloud Run IAP** (`iap_enabled = true` in `terraform/cicd_and_secrets.tf`, `IAP_STRICT_MODE=true`), restricting `roles/run.invoker` exclusively to the Google-managed IAP service agent (`terraform/iap_and_sso.tf`) and enforcing group/domain-only `iap_authorized_principals` via Terraform variable validation.
  2. Run Cloud Run under a dedicated custom service account (`sec-analyst-sa`) with read-only dataset/bucket roles and a custom table-scoped append-only role (`secEdgarTelemetrySinkAppend`) on the 3 telemetry/audit tables only (`terraform/iam.tf`, `terraform/telemetry.tf`).
  3. Authenticate GitHub Actions via keyless Workload Identity Federation (`sec-edgar-cicd-sa`, `roles/run.developer`, no JSON keys).
- **Alternatives Considered:**
  1. *External Global Application Load Balancer + Cloud Armor + Serverless NEG + VPC-SC:* Ideal for high-QPS multi-region production, but adds fixed LB/VPC/perimeter operational overhead in a single-region sandbox environment.
  2. *Public Cloud Run with app-only JWT checks:* Weaker defense-in-depth than terminating authentication at GCP IAP before requests reach the container.
- **Consequences & Production Network Architecture:**
  - *Current:* Strong zero-trust identity boundary and least-privilege IAM with minimal infrastructure footprint; `ingress = "INGRESS_TRAFFIC_ALL"` is gated by IAP (`roles/run.invoker` restricted to the IAP service agent).
  - *Production Enterprise Upgrade:* Place a Global External Application Load Balancer + Cloud CDN + **Cloud Armor** (WAF + per-user rate limiting) in front of multi-region Cloud Run (`INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER`), route egress over **Direct VPC Egress** + **Private Service Connect** to private data/cache tiers, and enclose BigQuery, GCS, Firestore, and Vertex AI inside a **VPC Service Controls (VPC-SC)** perimeter.

---

## ADR-005: Split Storage Architecture (Firestore + BigQuery + GCS)

- **Status:** Accepted
- **Context:** Workloads span four distinct access patterns: analytical SQL/vector scans, append-only telemetry/audit logs, large immutable 10-K section blobs, and low-latency mutable per-user sessions/feedback.
- **Decision:** Match each workload to its native GCP store:
  - **BigQuery (`sec_edgar`):** Structured XBRL `financial_metrics`, 768-d `filing_chunks`, YoY/chunk-health SQL views, golden eval tables, and append-only telemetry/audit tables.
  - **Cloud Storage (`sec_corpus`):** Immutable full Item 1A / Item 7 Markdown sections for the source viewer and cached section reads, with uniform bucket-level access, public access prevention, versioning, and 7-day soft delete.
  - **Cloud Firestore (Native):** Mutable `analyst_sessions` (keyed by SHA-256 of `(user_id, session_id)` with transactional read-modify-write concurrency) and `analyst_feedback`.
- **Alternatives Considered:**
  1. *BigQuery for sessions and feedback:* Poor fit for low-latency point `GET`/`PATCH`/`DELETE` CRUD and subject to DML concurrency limits.
  2. *Cloud SQL / AlloyDB upfront:* Fixed always-on instance cost unnecessary for initial low-QPS deployment.
- **Consequences:**
  - *Pros:* Serverless scale-to-zero storage with clean read-only vs. mutable permission boundaries.
  - *Cons:* ADK `InMemorySessionService` event history and JWT revocation (`_REVOKED_TOKEN_FINGERPRINTS`) remain process-local in the current revision and must be externalized to a shared store (Firestore/Redis/Memorystore) when scaling horizontally across many instances.

---

## ADR-006: Server-Side Model Armor + Cloud DLP (SDP) with Fail-Closed Guardrails

- **Status:** Accepted
- **Context:** Financial AI assistants face prompt injection, credential/PII leakage, and regulatory risks around ungrounded forward-looking price targets, buy/sell recommendations, or confidential MNPI.
- **Decision:** Screen every prompt and response once at the orchestration boundary through **Google Cloud Model Armor** (`sec-edgar-analyst-armor`) configured with **server-side Cloud DLP (SDP)** inspect/de-identify templates (`terraform/model_armor.tf`) plus the deterministic **SEC Financial Projection & MNPI Policy Filter** (`projection_filter.py`, Git-versioned in `prompts/v1/model_armor_policy.yaml`). Default to **fail-closed** (`fail_closed = True`) if Model Armor is unreachable, and propagate the authenticated caller via `contextvars` (`bind_audit_principal`) so every guardrail block is attributed in `security_audit_logs`.
- **Alternatives Considered:**
  1. *Separate application-side Cloud DLP API calls + regex-only filters:* Adds extra RPC latency and weaker prompt-injection/RAI coverage.
  2. *Fail-open on guardrail outage:* Unacceptable compliance risk for regulated financial workflows.
- **Consequences:**
  - *Pros:* Single managed RPC combines prompt-injection, RAI, malicious URI, and PII de-identification; every block is auditable and trace-linked.
  - *Cons:* Regional Model Armor quota or availability issues will fail closed and block chat turns until resolved (mitigated via runbook triage and quota reservation in production).

---

## ADR-007: Agentic On-Demand Tool Selection with Turn-Scoped State Resets

- **Status:** Accepted
- **Context:** Pre-computing variance reports, anomaly flags, disclosure deltas, and sector map-reduce on every turn wasted latency/tokens and cluttered qualitative answers with unsolicited financial tables.
- **Decision:** Run only a lightweight hybrid-RAG evidence pre-pass (`_run_rag_prepass`) and let `FinancialAnalystAgent` invoke deterministic engines (`calculate_period_variances`, `flag_material_variances`, `compare_disclosures_across_years`, `sector_thematic_map_reduce`, `query_analytics_view`) on demand as ADK `FunctionTool`s. Reset all tool-output keys in ADK `session_state` at the start of every turn (`turn_scoped_state_seed`), hydrate outputs before claim verification (`hydrate_tool_outputs`), and cap per-turn retrieval calls via `retrieval_budget_before_tool_callback` (default 6).
- **Alternatives Considered:**
  1. *Deterministic regex intent router pre-computing all engines before LLM synthesis:* Brittle on compound queries and caused frequent false-positive UI cards.
  2. *Fully stateless single-turn sessions:* Loses multi-turn entity/window continuity and context caching benefits.
- **Consequences:**
  - *Pros:* Cleaner selective UI rendering (verified by the `selective_display_accuracy >= 0.90` eval gate), zero cross-turn payload leakage, and lower prompt prefill tokens on qualitative queries.
  - *Cons:* Complex multi-tool queries (e.g., anomaly + MD&A) require extra agent tool round-trips, increasing total wall-clock latency while keeping TTFT low.
