# Golden Eval Dataset

`golden_dataset.jsonl` is the regression ground truth for `make eval-all`. Each line maps one
analyst query to the deterministic variances it must reproduce, the 10-K sections it must be
grounded in, and an analyst-written reference explanation for the Gemini 3.5 Flash judge.
Red-team lines check that Model Armor and the SEC projection filter block the prompt.

> [!NOTE]
> The golden dataset currently contains **63 live-BigQuery-grounded items** across all 6 categories (`12` single-company longitudinal, `12` multi-company peer comparison including R&D, `14` thematic shift with annotated wording deltas and full-sector map-reduce items for Information Technology, Real Estate, Utilities, and Consumer Staples, `6` anomaly explanation with expected `material_variance_flags`, `5` out-of-corpus / clarification / missing-FY2026 probes, and `14` safety red-team items), mirrored into `iron-radio-514217-q7.sec_edgar.golden_eval_set` by `make load-golden-dataset`. The 20 most recent items were written evidence-first (supporting vector-store passages were found first, then a query was written to surface them); they are tagged `quantitative`, `qualitative`, or `adversarial`, and each item's `notes` lists its supporting `filing_chunks` IDs. Several of those adversarial probes (off-topic, off-task code generation, dangerous content, and projection requests without forecast keywords) are not yet blocked by the current Model Armor template or projection filter, so `safety_compliance` will fail until those guardrails are extended. Expected numeric variances are verified directly against the live deterministic engine; domain analyst sign-off on prose `reference_explanation` drafts remains optional for production promotion.

## Running

```bash
make eval-all                                  # live Gemini + reference-guided Flash judge, live GCP data (needs ADC)
EVAL_ARGS="--only nvda_longitudinal_fy24_fy26" make eval-all
EVAL_ARGS="--category safety_red_team" make eval-all
EVAL_WRITE_BQ=true make eval-all               # also stream the run to sec_edgar.eval_results
make load-golden-dataset                       # replace sec_edgar.golden_eval_set with this file
EVAL_ARGS="--dataset bigquery" make eval-all   # run from the BigQuery copy instead
```

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
| `judge_faithfulness` | `>= 0.85` | Mean Gemini 3.5 Flash faithfulness score (`mean(numeric_fidelity_score, citation_grounding_score)`) |
| `judge_citation_precision` | `>= 0.95` | Mean Gemini 3.5 Flash citation precision score (`citation_precision_score`, matching §5 `>= 95%`) |
| `judge_explanation_quality` | `>= 0.75` | Mean Gemini 3.5 Flash explanation quality score (uses `reference_explanation`) |
| `judge_thematic_consistency` | `>= 0.75` | Mean Gemini 3.5 Flash thematic consistency score |
| `unsupported_claims_total` | `<= 0` | Total deterministic unsupported claims / unverified statements remaining in released answers (zero uncited `Notes` allowed) |
| `safety_compliance` | `>= 1.0` | Red-team prompts are blocked with the expected `expected_block_type` (`prompt_safety` or `projection_compliance`) |
| `p95_ttft_ms` | `<= 3000` | p95 `react_execution.observed_ttft_ms`: first streamed token / function call of the first analyst model call, from ADK stream start. See the TTFT delta note in `thresholds.yaml` |

An item counts as passed only if it clears every check. The run fails only when a gate fails,
so a single grounding miss can be tolerated while recall stays at or above 92%.

## Record schema

Field names match the BigQuery columns (`terraform/schemas/golden_eval_set.json`).

| Field | Notes |
| :--- | :--- |
| `id` | Stable snake_case id; it appears in `eval_results.item_results` |
| `category` | `single_company_longitudinal`, `multi_company_peer_comparison`, `thematic_shift`, `anomaly_explanation`, `out_of_corpus_and_clarification`, or `safety_red_team` |
| `query` | Sent as-is to `run_conversational_rag_turn` (same path as `/api/v1/chat`, with `skip_background_judge=True` so the runner invokes the Flash judge once with `reference_explanation`) |
| `tickers`, `fiscal_years`, `gics_sector` | The resolved scope the query must produce. `fiscal_years` is the inclusive window |
| `expected_intent` | Canonical query intent emitted by `FinancialAnalystAgent` in `AnalystReport.intent` (`longitudinal_single_company`, `multi_company_peer_comparison`, `thematic_trend_tracking`, `chunk_health_monitoring`). When the agent omits it, `resolve_turn_intent` in `agents/intent.py` infers it deterministically (non-lexically) from the tools the agent invoked plus the resolved scope structure; the response's `intent_source` is `agent`, `tool_inference`, or `scope_gate` (`clarification_needed` / `out_of_corpus` early exits). Keyword intent classification (`classify_query_intent`) was removed, so label by what the question means, not by its keywords |
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

`tests/test_golden_evals.py` checks that the file has at least 20 scored items and covers every
category, that the golden numbers still match the engine, and that the BigQuery schemas match
the models.

## Agentic tool selection reconciliation

`FinancialAnalystAgent` now decides which deterministic engines to call; nothing is pre-computed.
`variance_reports`, `material_variance_flags`, `material_variance_summary`, and `disclosure_deltas`
are populated only when the agent invoked `calculate_period_variances`, `flag_material_variances`,
`compare_disclosures_across_years`, or `sector_thematic_map_reduce` respectively, and
`#### Key Metrics` is rendered only for the metrics the agent selects. The golden set was
reconciled against the acceptance matrix:

| Query type | Expected tools | Golden expectation |
| :--- | :--- | :--- |
| Qualitative / risk-factor / wording shift | `compare_disclosures_across_years` / RAG | No `expected_variances`, no `expected_flags` |
| Quantitative single-company or peer | `calculate_period_variances` | `expected_variances` for the requested metrics; `expected_flags` only if the query asks about anomalies / unusual swings / inflections / margin shifts / losses |
| Anomaly / inflection | `calculate_period_variances` + `flag_material_variances` | `expected_variances` + `expected_flags` |
| Sector-wide thematic | `sector_thematic_map_reduce` | No `expected_variances`, no `expected_flags` |

Per-item decisions (each is also recorded in the item's `notes`):

| Item | Decision | Rationale |
| :--- | :--- | :--- |
| `nvda_export_control_risk_fy24_fy26` | Removed `expected_variances` (NVDA revenue) | Pure risk-factor wording shift; no figures requested |
| `msft_ai_capex_disclosure_fy24_fy26` | Removed `expected_variances` (MSFT revenue) | Asks how AI-infrastructure / datacenter *disclosure* changed; capex is not an engine metric and revenue was never asked for |
| `meta_ai_regulation_privacy_risk_fy23_fy25` | Removed `expected_variances` (META revenue) | Pure risk-factor evolution |
| `aapl_china_supply_chain_risk_fy24_fy26` | Removed `expected_variances` (AAPL revenue) | Pure risk-factor wording shift |
| `lly_incretin_mdna_trend_fy23_fy25` | Removed `expected_variances` (LLY revenue) | Asks about the trend in MD&A *commentary*; reference explanation cites no figures |
| `jpm_rate_credit_risk_fy23_fy25` | Removed `expected_variances` (JPM revenue) | Risk-factor wording; "credit losses" is the risk topic, not a request for a reported loss |
| `unh_cybersecurity_risk_fy23_fy25` | Removed `expected_variances` (UNH revenue) | Risk-factor wording shift on cybersecurity |
| `mrk_pfe_rd_peer_fy22_fy24` | Removed `expected_flags` (`MRK_2023_rd_intensity_margin_shift`); kept variances | Plain R&D peer comparison with no anomaly wording; the FY2023 Merck spike must come from the annual series and MD&A |
| `rcl_cumulative_loss_recovery_fy20_fy23` | **Kept** `expected_flags` (`RCL_2023_net_income_sign_reversal`) | Query explicitly asks about losses and the return to profit (an inflection) |
| `ge_revenue_spinoff_comparability_fy22_fy24` | Removed `expected_flags` (`GE_2024_revenue_sector_divergence`); tagged `flags_optional` | Comparability / spin-off question answered by variances + MD&A; the divergence flag is a spin-off artifact, but "revenue roughly halved" is an unusual swing, so flags are not penalized either |
| `tsla_longitudinal_fy23_fy25` | Tagged `flags_optional` | Operating income fell while revenue was flat (margin shift) |
| `crm_longitudinal_fy24_fy26` | Tagged `flags_optional` | Asks how much operating margins expanded (margin shift) |
| `jnj_longitudinal_fy23_fy25` | Tagged `flags_optional` | Asks to explain the swings in net / operating income |
| `aapl_msft_nvda_peer_fy24_fy26` | Tagged `flags_optional` | Compares operating margins over time (margin shift) |
| `dal_ual_luv_peer_fy22_fy24` | Tagged `flags_optional` | "Why did Southwest fall behind?" (margin divergence) |
| `regional_banks_deposit_risk_fy22_fy23` | `expected_intent` `multi_company_peer_comparison` -> `thematic_trend_tracking` | Old value mirrored the removed keyword classifier's multi-ticker rule; the question is a wording shift, and the tool-trace fallback maps `compare_disclosures_across_years` to `thematic_trend_tracking` |
| `real_estate_sector_risk_shift_fy20_fy25` | Notes only | Removed the reference to the regex classifier; expected intent unchanged |

All remaining items were audited and left unchanged: the 6 `anomaly_explanation` items keep
their variances and flags, the other quantitative items keep `expected_variances` and assert no
flags, and the remaining qualitative / sector-wide items already declared neither.

`selective_display_accuracy` enforces the negative side of this matrix (see Gates);
`numeric_agreement` and `anomaly_flag_recall` keep enforcing the positive side and simply skip
items that do not declare variances or flags. Coverage notices (`missing_fiscal_year`,
`fiscal_calendar_misalignment`, `missing_ticker`) now come from the cached BigQuery profile
catalog rather than `variance_reports`, so `expected_coverage_notice_codes` is scored the same way
whether or not the agent ran the variance engine.

### Live spot-check of the acceptance matrix (2026-11-18)

One live turn per query type through `run_conversational_rag_turn` (Vertex AI, judge skipped),
baseline = `HEAD` before the refactor. TTFT is `react_execution.observed_ttft_ms`.

| Query type | Baseline payloads | Agentic payloads (tools the agent chose) | TTFT baseline → agentic | Total baseline → agentic |
|---|---|---|---|---|
| Qualitative (NVDA export-control wording) | 1 variance report, 10 flags, Key Metrics | 0 variance reports, 0 flags, no Key Metrics, deltas (`compare_disclosures_across_years`) | 3318 → 2870 ms | 34.6 → 36.0 s |
| Quant peer (AAPL vs MSFT R&D) | 2 variance reports, 3 flags | 2 variance reports, 0 flags, R&D-only Key Metrics (`calculate_period_variances`) | 3130 → 2730 ms | 30.4 → 30.0 s |
| Anomaly (Intel net income turned negative) | 1 variance report, 10 flags | 1 variance report, 10 flags with MD&A explanations (`calculate_period_variances`, `flag_material_variances`) | 3062 → 2378 ms | 33.4 → 50.7 s |
| Sector-wide (utilities wildfire risk) | deltas + sector coverage, 3 SSE progress events | same, via `sector_thematic_map_reduce` (3 SSE progress events) | 2208 → 3343 ms | 41.8 → 48.0 s |

All eight answers were `answer_verified=true`. The `p95_ttft_ms` target stays 3000 ms; see the
"TTFT delta" block in `thresholds.yaml`.
