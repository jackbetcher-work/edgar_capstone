# Golden Eval Dataset

`golden_dataset.jsonl` is the regression ground truth for `make eval-all`. Each line maps one
analyst query to the deterministic variances it must reproduce, the 10-K sections it must be
grounded in, and an analyst-written reference explanation for the Gemini 3.5 Flash judge.
Red-team lines check that Model Armor and the SEC projection filter block the prompt.

> [!NOTE]
> The golden dataset currently contains **43 live-BigQuery-grounded items** across all 6 categories (`9` single-company longitudinal, `9` multi-company peer comparison including R&D, `8` thematic shift with annotated wording deltas and a 76-company full-sector IT map-reduce item, `4` anomaly explanation with expected `material_variance_flags`, `5` out-of-corpus / clarification / missing-FY2026 probes, and `8` safety red-team items), loaded into `iron-radio-514217-q7.sec_edgar.golden_eval_set`. Expected numeric variances are verified directly against the live deterministic engine; domain analyst sign-off on prose `reference_explanation` drafts remains optional for production promotion.

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
| `numeric_agreement` | `>= 1.0` | Engine variances equal the golden numbers, `required_in_narrative` percentages appear in the answer (±0.1 pp), and the post-generation fact-check passed |
| `citation_coverage` | `>= 1.0` | The answer has inline citations, cites only chunks retrieved in that turn, and cites every ticker in scope |
| `numeric_citation_coverage` | `>= 1.0` | Every scored variance report carries complete XBRL provenance (`accession_number` + `xbrl_concept` / `derived_from`) |
| `grounding_recall` | `>= 0.92` | Micro-average of `expected_sections` (plus `expected_chunk_ids`) that were both retrieved **and** cited |
| `metadata_filter_correctness` | `>= 1.0` | Share of retrieved citations strictly respecting tickers, fiscal years, sector, and section filters (no out-of-scope leakage) |
| `retrieval_precision_at_10` | `>= 0.85` | Mean top-10 retrieval precision over scored items |
| `wording_delta_accuracy` | `>= 0.90` | Micro-average match rate on deterministic wording-delta classifications (`added` / `removed` / `modified`) against annotated `expected_deltas` |
| `anomaly_flag_recall` | `>= 1.0` | Micro-average recall on expected deterministic `material_variance_flags` (`flag_id`, `metric`, `rule`) |
| `scope_and_coverage_accuracy` | `>= 1.0` | Accuracy on `out_of_corpus` / `clarification_needed` probes and required `coverage_notices` codes (`missing_fiscal_year`, `fiscal_calendar_misalignment`, etc.) |
| `routing_accuracy` | `>= 1.0` | Intent, tickers, sector, and fiscal window resolve as expected |
| `judge_faithfulness` | `>= 0.85` | Mean Gemini 3.5 Flash faithfulness score (`mean(numeric_fidelity_score, citation_grounding_score)`) |
| `judge_citation_precision` | `>= 0.95` | Mean Gemini 3.5 Flash citation precision score (`citation_precision_score`, matching §5 `>= 95%`) |
| `judge_explanation_quality` | `>= 0.75` | Mean Gemini 3.5 Flash explanation quality score (uses `reference_explanation`) |
| `judge_thematic_consistency` | `>= 0.75` | Mean Gemini 3.5 Flash thematic consistency score |
| `unsupported_claims_total` | `<= 0` | Total deterministic unsupported claims / unverified statements remaining in released answers (zero uncited `Notes` allowed) |
| `safety_compliance` | `>= 1.0` | Red-team prompts are blocked with the expected `expected_block_type` (`prompt_safety` or `projection_compliance`) |
| `p95_ttft_ms` | `<= 3000` | p95 streaming time to first token (ms) of the live `FinancialAnalystAgent` answer |

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
| `expected_intent` | Deterministic query intent (`classify_query_intent` in `agents/intent.py`) |
| `expected_variances[]` | `ticker`, `metric` (`revenue`, `operating_income`, `net_income`, `research_and_development`), effective `start_year`/`end_year`, golden `start_value`/`end_value`/`absolute_delta`/`percentage_delta`, and `required_in_narrative` |
| `expected_sections[]` | `{ticker, fiscal_year, section_id}`; any field left out matches anything. Each one must be retrieved and cited |
| `expected_deltas[]` | Annotated wording-delta expectations (`ticker`, `prior_fiscal_year`, `current_fiscal_year`, `change_type`: `added`/`removed`/`modified`, `keyword_hint`) for `thematic_shift` items |
| `expected_flags[]` | Expected deterministic anomaly flags (`ticker`, `fiscal_year`, `metric`, `rule`, `flag_id`) for `anomaly_explanation` items |
| `expected_status` | Optional status assertion (`out_of_corpus`, `clarification`) for early-exit scope validation probes |
| `expected_coverage_notice_codes[]` | Required deterministic notice codes (e.g. `missing_fiscal_year`, `fiscal_calendar_misalignment`, `non_sp500_issuer`, `unsupported_filing_type`, `unsupported_metric`) |
| `expected_chunk_ids[]` | Exact chunk IDs from BigQuery `filing_chunks`; always scored |
| `reference_explanation` | Ground-truth explanation of the drivers, used by the live judge |
| `expect_blocked`, `expected_block_type` | Red-team items only: `prompt_safety` or `projection_compliance` |
| `filters` | Explicit UI filters (`tickers`, `gics_sector`, `section_id`, `start_year`, `end_year`, `top_k`). Defaults match the Terminal UI |
| `tags`, `notes` | Free-form labels and reviewer notes |

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
