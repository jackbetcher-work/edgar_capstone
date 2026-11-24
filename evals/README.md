# Golden Eval Dataset

`golden_dataset.jsonl` is the regression ground truth for `make eval-all`. Each line maps one
analyst query to the deterministic variances it must reproduce, the 10-K sections it must be
grounded in, and an analyst-written reference explanation for the Gemini 3.8 Flash (`ThinkingLevel.LOW`) judge.
Red-team lines check that Model Armor and the SEC projection filter block the prompt.

> [!NOTE]
> The golden dataset contains **30 live-BigQuery-grounded items** across 6 categories (`6` single-company longitudinal, `6` multi-company peer comparison including R&D, `7` thematic shift with annotated wording deltas and full-sector map-reduce items for Information Technology, Utilities, and Consumer Staples, `3` anomaly explanation with expected `material_variance_flags`, `4` out-of-corpus / clarification / missing-FY2026 probes, and `4` safety red-team items), mirrored into `iron-radio-514217-q7.sec_edgar.golden_eval_set` by `make load-golden-dataset`. Expected numeric variances are verified against the deterministic engine, and evidence-first items (tagged `quantitative`, `qualitative`, or `adversarial`) list their supporting `filing_chunks` IDs in `notes`.

## Running

```bash
make eval-all                                  # live Gemini + Flash judge, concurrent pool (default 4 workers)
EVAL_CONCURRENCY=1 make eval-all               # sequential run; enforces the p95_ttft_ms gate
EVAL_CONCURRENCY=6 make eval-all               # custom worker pool size (higher values risk 429s / higher TTFT)
EVAL_ARGS="--only nvda_longitudinal_fy24_fy26" make eval-all
EVAL_ARGS="--category safety_red_team" make eval-all
EVAL_WRITE_BQ=true make eval-all               # also stream the run to sec_edgar.eval_results
make load-golden-dataset                       # replace sec_edgar.golden_eval_set with this file
EVAL_ARGS="--dataset bigquery" make eval-all   # run from the BigQuery copy instead
```

> [!NOTE]
> **Concurrency & `p95_ttft_ms`:** By default, `scripts/run_evals.py` evaluates up to `4` golden
> items concurrently (`--concurrency 4` / `EVAL_CONCURRENCY=4`), executing synchronous GCP
> pre-pass I/O across a bounded worker pool while multiplexing live ADK Gemini streams
> asynchronously onto `_ADK_LOOP_MANAGER`. Concurrent requests (especially alongside the three
> sector map-reduce items, which spawn their own parallel Flash batches) can **increase
> `p95_ttft_ms`** due to client/event-loop contention and Vertex AI request queueing. Whenever
> effective concurrency exceeds `1`, `p95_ttft_ms` is still measured and recorded in the JSON
> report and BigQuery, but the `p95_ttft_ms` gate is automatically marked **`SKIP`** so
> pool-induced latency does not cause false regressions. Run with `EVAL_CONCURRENCY=1 make eval-all`
> (or `--concurrency 1`) for an isolated latency benchmark that enforces `p95_ttft_ms <= 3000`.

Exit codes: `0` all gates passed, `1` a gate regressed, `2` invalid dataset, `3` BigQuery
read/write failure. Each run writes `build/evals/eval_report_<run_id>.json` (and `latest.json`).

## Gates (`thresholds.yaml`)

| Gate | Threshold | What it measures |
| :--- | :--- | :--- |
| `numeric_agreement` | `>= 1.0` | Engine variances equal the golden numbers, `required_in_narrative` percentages appear in the answer (±0.1 pp), and the post-generation fact-check passed. Items without `expected_variances` are not scored |
| `citation_coverage` | `>= 1.0` | The answer has inline citations, cites only chunks retrieved in that turn, and cites every ticker in scope |
| `numeric_citation_coverage` | `>= 1.0` | Every scored variance report carries complete XBRL provenance (`accession_number` + `xbrl_concept` / `derived_from`) |
| `grounding_recall` | `>= 0.92` | Micro-average of `expected_sections` (plus `expected_chunk_ids`) that were both retrieved **and** cited |
| `metadata_filter_correctness` | `>= 1.0` | Share of retrieved citations strictly respecting tickers, fiscal years, sector, and section filters (no out-of-scope leakage) |
| `retrieval_precision_at_10` | `>= 0.85` | Mean top-10 retrieval precision over scored items |
| `wording_delta_accuracy` | `>= 0.90` | Micro-average match rate on deterministic wording-delta classifications (`added` / `removed` / `modified`) against annotated `expected_deltas` |
| `anomaly_flag_recall` | `>= 1.0` | Micro-average recall on expected deterministic `material_variance_flags` (`flag_id`, `metric`, `rule`); items without `expected_flags` contribute nothing |
| `selective_display_accuracy` | `>= 0.90` | Share of successful scored items where (a) items with no `expected_variances` returned empty `variance_reports` and no `#### Key Metrics`, and (b) items with no `expected_flags` (and no `flags_optional` tag) returned empty `material_variance_flags`. Below 1.0 to tolerate live Gemini tool-selection nondeterminism |
| `scope_and_coverage_accuracy` | `>= 1.0` | Accuracy on `out_of_corpus` / `clarification_needed` probes and required `coverage_notices` codes (`missing_fiscal_year`, `fiscal_calendar_misalignment`, etc.) |
| `routing_accuracy` | `>= 1.0` | Intent (agent-emitted, see `expected_intent`), tickers, sector, and fiscal window resolve as expected |
| `judge_faithfulness` | `>= 0.85` | Mean Gemini 3.8 Flash (low-thinking) faithfulness score (`mean(numeric_fidelity_score, citation_grounding_score)`) |
| `judge_citation_precision` | `>= 0.95` | Mean Gemini 3.8 Flash (low-thinking) citation precision score (`citation_precision_score`) |
| `judge_explanation_quality` | `>= 0.75` | Mean Gemini 3.8 Flash (low-thinking) explanation quality score (uses `reference_explanation`) |
| `judge_thematic_consistency` | `>= 0.75` | Mean Gemini 3.8 Flash (low-thinking) thematic consistency score |
| `unsupported_claims_total` | `<= 0` | Total deterministic unsupported claims or unverified statements remaining in released answers |
| `safety_compliance` | `>= 1.0` | Red-team prompts are blocked with the expected `expected_block_type` (`prompt_safety` or `projection_compliance`) |
| `p95_ttft_ms` | `<= 3000` | p95 `react_execution.observed_ttft_ms`: first streamed token / function call of the first analyst model call, from ADK stream start. Automatically marked `SKIP` (while still recording the observed value) when effective concurrency `> 1`, because concurrent requests can inflate TTFT; use `--concurrency 1` to enforce. See `thresholds.yaml` |

An item counts as passed only if it clears every check. The run fails only when a gate fails,
so a single grounding miss can be tolerated while recall stays at or above 92%.

## Record schema

Field names match the BigQuery columns (`terraform/schemas/golden_eval_set.json`).

| Field | Notes |
| :--- | :--- |
| `id` | Stable snake_case id; it appears in `eval_results.item_results` |
| `category` | `single_company_longitudinal`, `multi_company_peer_comparison`, `thematic_shift`, `anomaly_explanation`, `out_of_corpus_and_clarification`, or `safety_red_team` |
| `query` | Sent as-is to `run_conversational_rag_turn` (same path as `/api/v1/chat`); the eval runner then invokes the Flash judge once synchronously with `reference_explanation` |
| `tickers`, `fiscal_years`, `gics_sector` | The resolved scope the query must produce. `fiscal_years` is the inclusive window |
| `expected_intent` | Canonical query intent emitted by `FinancialAnalystAgent` in `AnalystReport.intent` (`longitudinal_single_company`, `multi_company_peer_comparison`, `thematic_trend_tracking`, `chunk_health_monitoring`). When omitted, `resolve_turn_intent` in `agents/orchestration/intent.py` infers it deterministically from the tools invoked and resolved scope; `intent_source` is `agent`, `tool_inference`, or `scope_gate` (`clarification_needed` / `out_of_corpus` early exits). Label by what the question asks, not by surface keywords |
| `expected_variances[]` | Declare only when the query asks for figures: an item without them asserts that the agent does **not** call `calculate_period_variances` (no `variance_reports`, no Key Metrics). `ticker`, `metric` (`revenue`, `operating_income`, `net_income`, `research_and_development`), effective `start_year`/`end_year`, golden `start_value`/`end_value`/`absolute_delta`/`percentage_delta`, and `required_in_narrative` |
| `expected_sections[]` | `{ticker, fiscal_year, section_id}`; any field left out matches anything. Each one must be retrieved and cited |
| `expected_deltas[]` | Annotated wording-delta expectations (`ticker`, `prior_fiscal_year`, `current_fiscal_year`, `change_type`: `added`/`removed`/`modified`, `keyword_hint`) for `thematic_shift` items |
| `expected_flags[]` | Expected deterministic anomaly flags (`ticker`, `fiscal_year`, `metric`, `rule`, `flag_id`). Declare only when the query asks about anomalies, unusual swings, inflections, margin shifts, or losses; an item without them (and without the `flags_optional` tag) asserts empty `material_variance_flags` |
| `expected_status` | Optional status assertion (`out_of_corpus`, `clarification`) for early-exit scope validation probes |
| `expected_coverage_notice_codes[]` | Required deterministic notice codes (e.g. `missing_fiscal_year`, `fiscal_calendar_misalignment`, `non_sp500_issuer`, `unsupported_filing_type`, `unsupported_metric`) |
| `expected_chunk_ids[]` | Exact chunk IDs from BigQuery `filing_chunks`; always scored |
| `reference_explanation` | Ground-truth explanation of the drivers, used by the live judge |
| `expect_blocked`, `expected_block_type` | Red-team items only: `prompt_safety` or `projection_compliance` |
| `filters` | Explicit UI filters (`tickers`, `gics_sector`, `section_id`, `start_year`, `end_year`, `top_k`). Defaults match the Terminal UI |
| `tags`, `notes` | Free-form labels and reviewer notes. The `flags_optional` tag marks items whose query permits `flag_material_variances` without asserting specific flags (cannot be combined with `expected_flags`) |

## Adding or changing an item

1. Add a line with the query, scope, `expected_sections`, `reference_explanation`, and the
   variances to check. Leave the numeric fields `null`.
2. Run `make refresh-golden-expectations` to fill in the numbers from the engine, then **review the
   diff**. Every variance uses the item's `fiscal_years` window. Never refresh just to make a
   failing run pass, because that would hide a real numeric regression.
3. Run `make eval-all` (live GCP data and live Gemini; requires valid Application Default
   Credentials).
4. Run `make load-golden-dataset` so BigQuery matches the committed file. `dataset_version` (a
   content hash) links each `eval_results` row to the exact golden set it ran against.

`tests/test_golden_evals.py` checks that the file has at least 20 scored items, meets the per-category floors (`6` single-company, `6` peer, `7` thematic, `3` anomaly, `4` out-of-corpus/clarification, `4` safety), keeps golden numbers in sync with the engine, and matches the BigQuery schemas.

## Dataset coverage

The 30-item dataset spans semiconductors, hardware, energy, staples, industrials, airlines, pharma, retail, advertising, media, banks, cybersecurity, and utilities:

- **Core workflows:** R&D peer comparison (`nvda_aapl_rd_peer_fy24_fy26`) and sector-wide AI risk wording shifts (`it_sector_genai_risk_trend_fy24_fy26`).
- **Analyst edge cases:** spin-off comparability (`ge_*`), cumulative loss aggregation (`rcl_*`), CAGR with a misspelled entity (`pg_*`), fiscal-calendar misalignment (`nvda_aapl_*`, `crwd_*`), and margin compression on flat revenue (`tsla_*`).
- **Anomaly flags:** sign reversals across Intel, Micron, Boeing, and Royal Caribbean (9 `expected_flags`).
- **Wording deltas and sector map-reduce:** NVIDIA export controls, CrowdStrike July 19 outage, 3M PFAS / earplug litigation (12 `expected_deltas`), regional bank deposit risk after SVB, and three full-sector map-reduce reviews (Information Technology, Utilities, Consumer Staples).

## Tool selection expectations

`FinancialAnalystAgent` chooses which deterministic tools to invoke per query. `variance_reports`, `material_variance_flags`, `material_variance_summary`, and `disclosure_deltas` are populated only when the agent calls `calculate_period_variances`, `flag_material_variances`, `compare_disclosures_across_years`, or `sector_thematic_map_reduce` respectively, and `#### Key Metrics` renders only for the selected metrics.

| Query type | Expected tools | Golden expectation |
| :--- | :--- | :--- |
| Qualitative / risk-factor / wording shift | `compare_disclosures_across_years` / RAG | No `expected_variances`, no `expected_flags` |
| Quantitative single-company or peer | `calculate_period_variances` | `expected_variances` for the requested metrics; `expected_flags` only if the query asks about anomalies, unusual swings, inflections, margin shifts, or losses |
| Anomaly / inflection | `calculate_period_variances` + `flag_material_variances` | `expected_variances` + `expected_flags` |
| Sector-wide thematic | `sector_thematic_map_reduce` | No `expected_variances`, no `expected_flags` |

`selective_display_accuracy` checks that variance and anomaly tools are not over-invoked when variances or flags are not expected, while `numeric_agreement` and `anomaly_flag_recall` score items that declare `expected_variances` or `expected_flags`. Coverage notices (`missing_fiscal_year`, `fiscal_calendar_misalignment`, `missing_ticker`) are derived from the cached BigQuery profile catalog, so `expected_coverage_notice_codes` is evaluated independently of whether the variance engine ran.
