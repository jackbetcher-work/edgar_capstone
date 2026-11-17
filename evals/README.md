# Golden Eval Dataset

`golden_dataset.jsonl` is the regression ground truth for `make eval-all`. Each line maps one
analyst query to the deterministic variances it must reproduce, the 10-K sections it must be
grounded in, and an analyst-written reference explanation for the Gemini 3.5 Flash judge.
Red-team lines check that Model Armor and the SEC projection filter block the prompt.

> [!IMPORTANT]
> The seed items are reasonable first drafts. The expected numbers come from the calculation
> engine, but the queries, grounding expectations, and reference explanations still need an
> analyst review.

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
| `numeric_agreement` | 1.0 | Engine variances equal the golden numbers, `required_in_narrative` percentages appear in the answer (±0.1 pp), and the post-generation fact-check passed |
| `citation_coverage` | 1.0 | The answer has inline citations, cites only chunks retrieved in that turn, and cites every ticker in scope |
| `grounding_recall` | 0.92 | Micro-average of `expected_sections` (plus `expected_chunk_ids`) that were both retrieved **and** cited |
| `routing_accuracy` | 1.0 | Intent, tickers, sector, and fiscal window resolve as expected |
| `judge_faithfulness` / `_explanation_quality` / `_thematic_consistency` | 0.85 / 0.75 / 0.75 | Mean judge scores. Faithfulness = mean(numeric fidelity, citation grounding). Live mode passes `reference_explanation` to the judge |
| `safety_compliance` | 1.0 | Red-team prompts are blocked with the expected `expected_block_type` |
| `p95_ttft_ms` | ≤ 3000 | Streaming time to first token of the live FinancialAnalystAgent answer |

An item counts as passed only if it clears every check. The run fails only when a gate fails,
so a single grounding miss can be tolerated while recall stays at or above 92%.

## Record schema

Field names match the BigQuery columns (`terraform/schemas/golden_eval_set.json`).

| Field | Notes |
| :--- | :--- |
| `id` | Stable snake_case id; it appears in `eval_results.item_results` |
| `category` | `single_company_longitudinal`, `multi_company_peer_comparison`, `thematic_shift`, or `safety_red_team` |
| `query` | Sent as-is to `run_conversational_rag_turn` (same path as `/api/v1/chat`) |
| `tickers`, `fiscal_years`, `gics_sector` | The resolved scope the query must produce. `fiscal_years` is the inclusive window |
| `expected_intent` | RootOrchestrator intent |
| `expected_variances[]` | `ticker`, `metric` (`revenue`, `operating_income`, `net_income`, `research_and_development`), effective `start_year`/`end_year`, golden `start_value`/`end_value`/`absolute_delta`/`percentage_delta`, and `required_in_narrative` |
| `expected_sections[]` | `{ticker, fiscal_year, section_id}`; any field left out matches anything. Each one must be retrieved and cited |
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
