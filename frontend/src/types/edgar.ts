export type SectionFilterId = 'ALL' | 'item_1a' | 'item_7'

export type QueryIntentType =
  | 'longitudinal_single_company'
  | 'multi_company_peer_comparison'
  | 'thematic_trend_tracking'
  | 'chunk_health_monitoring'
  | 'safety_guardrail_block'
  | 'projection_compliance_block'
  | 'clarification_needed'
  | 'out_of_corpus'

export interface CompanySummary {
  ticker: string
  company_name: string
  cik: string
  gics_sector: string
  gics_sub_industry: string
  fiscal_years: number[]
  latest_fiscal_year: number
  latest_revenue: number | null
  latest_operating_income: number | null
  latest_net_income: number | null
  latest_research_and_development?: number | null
  latest_operating_margin_pct: number | null
  latest_net_margin_pct: number | null
  sec_filing_url: string
}

/** Load state of the backend `/api/v1/companies` catalog. */
export type CatalogLoadStatus = 'loading' | 'ready' | 'error'

export interface CompaniesCatalogResponse {
  status: string
  company_count: number
  total_profiles: number
  min_year: number
  max_year: number
  gics_sectors: string[]
  companies: CompanySummary[]
}

export interface NumericCitation {
  ticker: string
  fiscal_year: number
  metric: string
  value?: number | null
  unit?: string
  period_of_report?: string | null
  filing_date?: string | null
  accession_number?: string | null
  xbrl_concept?: string | null
  sec_filing_url?: string | null
  derived_from?: NumericCitation[]
}

export interface CoverageNotice {
  code: string
  severity: 'info' | 'warning'
  message: string
  tickers?: string[]
  missing_fiscal_years?: number[]
}

export interface SectorCoverageMetadata {
  gics_sector?: string
  total_sector_companies?: number
  companies_analyzed?: number
  companies_with_matches?: number
  chunks_analyzed?: number
  batches_executed?: number
  batches_succeeded?: number
  batch_size?: number
  theme_terms?: string[]
  section_id?: string
  start_year?: number
  end_year?: number
  map_model?: string
}

export interface MetricVariance {
  metric_name: 'revenue' | 'operating_income' | 'net_income' | 'research_and_development'
  start_year: number
  end_year: number
  start_value: number | null
  end_value: number | null
  absolute_delta: number | null
  percentage_delta: number | null
  cagr_pct: number | null
  direction: 'increase' | 'decrease' | 'flat' | 'unavailable'
  start_period_of_report?: string | null
  end_period_of_report?: string | null
  start_accession_number?: string | null
  end_accession_number?: string | null
  start_sec_filing_url?: string | null
  end_sec_filing_url?: string | null
  start_xbrl_concept?: string | null
  end_xbrl_concept?: string | null
  derived_from?: NumericCitation[]
}

export interface AnnualStepVariance {
  fiscal_year: number
  prev_fiscal_year: number | null
  period_of_report?: string | null
  filing_date?: string | null
  accession_number?: string | null
  prior_period_of_report?: string | null
  prior_accession_number?: string | null
  prior_sec_filing_url?: string | null
  metric_provenance?: Record<string, string | null> | null
  revenue: number | null
  revenue_yoy_delta: number | null
  revenue_yoy_pct: number | null
  operating_income: number | null
  operating_income_yoy_delta: number | null
  operating_income_yoy_pct: number | null
  net_income: number | null
  net_income_yoy_delta: number | null
  net_income_yoy_pct: number | null
  research_and_development?: number | null
  research_and_development_yoy_delta?: number | null
  research_and_development_yoy_pct?: number | null
  sec_filing_url: string | null
}

export interface CompanyVarianceReport {
  ticker: string
  company_name: string
  cik: string
  gics_sector: string
  gics_sub_industry: string
  start_year: number
  end_year: number
  requested_start_year?: number | null
  requested_end_year?: number | null
  available_fiscal_years?: number[]
  missing_fiscal_years?: number[]
  currency: string
  revenue_variance: MetricVariance
  operating_income_variance: MetricVariance
  net_income_variance: MetricVariance
  /** Absent in older payloads; direction 'unavailable' when the issuer reports no R&D line. */
  research_and_development_variance?: MetricVariance | null
  operating_margin_bps_delta: number | null
  net_margin_bps_delta: number | null
  rd_intensity_bps_delta?: number | null
  annual_series: AnnualStepVariance[]
  source_urls: Record<string, string>
  period_end_dates?: Record<string, string>
  filing_dates?: Record<string, string>
  accession_numbers?: Record<string, string>
  metric_provenance?: Record<string, Record<string, string | null>>
  derived_margin_inputs?: Record<string, NumericCitation[]>
}

export interface SectorBenchmarkRow {
  gics_sector: string
  fiscal_year: number
  company_count: number
  tickers: string[]
  total_sector_revenue: number | null
  avg_company_revenue: number | null
  total_sector_operating_income: number | null
  total_sector_net_income: number | null
  total_sector_research_and_development?: number | null
}

export interface GroundedCitation {
  chunk_id: string
  profile_id: string
  ticker: string
  company_name: string
  gics_sector: string
  fiscal_year: number
  section_id: 'item_1a' | 'item_7'
  section_title: string
  excerpt: string
  sec_filing_url: string
  unstructured_gcs_uri: string
  score: number
  period_of_report?: string | null
  filing_date?: string | null
  accession_number?: string | null
}

export interface FactCheckVerification {
  passed: boolean
  checked_metrics_count: number
  matched_citations_count: number
  has_required_citations: boolean
  discrepancies: string[]
  model_latency_ms: number
}

export interface AgentTraceStep {
  step: number
  agent: string
  agent_service_account?: string
  spiffe_id?: string
  s2s_target_service?: string
  s2s_verified?: boolean
  tool: string
  status: 'completed' | 'warning' | 'blocked'
  duration_ms: number
  summary: string
}

export interface AgentIdentitySummary {
  agent_name: string
  service_account_email: string
  spiffe_id: string
}

export interface ContextCacheInfo {
  cache_key: string
  cache_hit: boolean
  ticker: string
  fiscal_year: number
  section_id: 'item_1a' | 'item_7'
  token_estimate: number
  sec_filing_url: string
}

export interface ResolvedQueryScope {
  tickers: string[]
  tickers_csv?: string
  gics_sector: string
  section_id: string
  start_year: number
  end_year: number
  memory_inherited?: boolean
}

export interface ReActStepMetadata {
  iteration: number
  phase: string
  thought: string
  action: string
  observation: string
  agent: string
  model_used?: string | null
  context_tokens: number
  cached_context_tokens: number
  context_latency_ms: number
  duration_ms: number
  status?: string
}

export interface NarrativeEvaluationMetadata {
  evaluator_model: string
  task_category: string
  grade: 'pass' | 'needs_revision' | 'fail'
  overall_score: number
  numeric_fidelity_score: number
  citation_grounding_score: number
  explanation_quality_score: number
  thematic_consistency_score: number
  rationale: string
  context_tokens: number
  context_latency_ms: number
  total_latency_ms: number
}

export interface TurnContextLatencySummary {
  total_calls: number
  total_context_tokens: number
  total_cached_context_tokens: number
  total_uncached_context_tokens: number
  mean_context_latency_ms: number
  p50_context_latency_ms: number
  p95_context_latency_ms: number
  mean_total_model_latency_ms: number
  p95_ttft_ms: number
  ttft_slo_target_ms: number
  ttft_slo_compliance_ratio: number
  total_estimated_cache_latency_saved_ms: number
}

export interface MaterialVarianceFlag {
  flag_id: string
  ticker: string
  company_name: string
  gics_sector: string
  fiscal_year: number
  prior_fiscal_year: number | null
  metric:
    | 'revenue'
    | 'operating_income'
    | 'net_income'
    | 'research_and_development'
    | 'operating_margin'
    | 'net_margin'
    | 'rd_intensity'
  rule:
    | 'yoy_change'
    | 'sign_reversal'
    | 'margin_shift'
    | 'period_change'
    | 'sector_divergence'
  severity: 'high' | 'medium'
  value: number
  unit: 'pct' | 'bps' | 'pp' | 'usd'
  threshold: number
  prior_value: number | null
  current_value: number | null
  benchmark_value: number | null
  peer_count: number | null
  description: string
  explanation_query: string
  source: {
    accession_number: string
    sec_filing_url: string
    period_of_report: string
    xbrl_concept: string | null
    prior_accession_number: string | null
    prior_sec_filing_url: string | null
    prior_period_of_report: string | null
    prior_xbrl_concept?: string | null
    denominator_xbrl_concept?: string | null
    prior_denominator_xbrl_concept?: string | null
  }
  explanation: {
    chunk_id: string
    section_id: 'item_1a' | 'item_7'
    section_title: string
    excerpt: string
    sec_filing_url: string
  } | null
}

export interface MaterialVarianceSummary {
  total_flags: number
  returned_flags: number
  high_severity_count: number
  explained_flags: number
  thresholds: Record<string, number>
}

export type DisclosureChangeType = 'added' | 'removed' | 'modified' | 'unchanged'

export interface DisclosureDelta {
  delta_id: string
  ticker: string
  company_name: string
  section_id: 'item_1a' | 'item_7'
  section_title: string
  change_type: DisclosureChangeType
  prior_fiscal_year: number
  current_fiscal_year: number
  prior_chunk_id: string | null
  current_chunk_id: string | null
  prior_excerpt: string | null
  current_excerpt: string | null
  similarity: number
  added_phrases: string[]
  removed_phrases: string[]
  theme_relevance: number
}

export interface DisclosureYearPair {
  prior_fiscal_year: number
  current_fiscal_year: number
  counts: Partial<Record<DisclosureChangeType, number>>
  deltas: DisclosureDelta[]
}

export interface CompanyDisclosureEvolution {
  ticker: string
  company_name: string
  gics_sector: string
  fiscal_years_available: number[]
  missing_fiscal_years: number[]
  theme_mentions: Record<string, number>
  year_pairs: DisclosureYearPair[]
}

export interface DisclosureDeltaResult {
  status: 'success' | 'error'
  message?: string
  theme?: string
  theme_terms?: string[]
  section_id?: SectionFilterId
  start_year?: number
  end_year?: number
  companies_selected_by?: 'explicit' | 'theme_relevance'
  summary?: {
    added: number
    removed: number
    modified: number
    unchanged: number
    year_pairs: number
    companies_analyzed: number
  }
  companies?: CompanyDisclosureEvolution[]
  citations?: GroundedCitation[]
}

/**
 * SSE progress event. `stage` is open-ended: known stages include
 * `scope_resolved` (meta.intent may be null — the agent decides intent later),
 * `agent_tool` (meta: { tool: string }) and `sector_map`; unknown stages are
 * displayed by their `message` like any other.
 */
export interface ChatProgressEvent {
  stage: string
  message: string
  meta?: Record<string, unknown> | null
  elapsed_ms?: number
}

/** Who determined the turn intent (agent tool calls vs. turn-gate scope checks). */
export type IntentSource = 'agent' | 'tool_inference' | 'scope_gate'

export interface ChatTurnResponse {
  status: 'success' | 'blocked' | 'clarification' | 'out_of_corpus' | 'error'
  model_armor_blocked: boolean
  intent: QueryIntentType
  intent_source?: IntentSource
  orchestrator: string
  resolved_scope: ResolvedQueryScope
  answer_markdown: string
  /** False when the live answer failed verification and its narrative was withheld. */
  answer_verified?: boolean
  clarification_reason?: string | null
  suggested_followups?: string[]
  coverage_notices?: CoverageNotice[]
  sector_coverage?: SectorCoverageMetadata | null
  missing_tickers?: string[]
  structured_report?: Record<string, unknown> | null
  report_verification?: {
    passed?: boolean
    findings_verified?: number
    findings_rejected?: number
    unsupported_claims_count?: number
  } | null
  // Deterministic engine outputs are agent-invoked tools: each is empty/null
  // unless the agent called the matching tool during the turn.
  /** `[]` unless the agent called `calculate_period_variances`. */
  variance_reports?: CompanyVarianceReport[]
  sector_benchmarks?: SectorBenchmarkRow[]
  /** `[]` unless the agent called `flag_material_variances`. */
  material_variance_flags?: MaterialVarianceFlag[]
  material_variance_summary?: MaterialVarianceSummary | null
  /** `null` unless the agent called `compare_disclosures_across_years` / sector map-reduce. */
  disclosure_deltas?: DisclosureDeltaResult | null
  /** May be empty (e.g. turn-gate / pure-XBRL turns). */
  citations: GroundedCitation[]
  candidates_searched?: number
  context_cache: ContextCacheInfo | null
  fact_check: FactCheckVerification
  evaluation?: NarrativeEvaluationMetadata | null
  react_execution?: {
    strategy: string
    iterations: number
    judge_mode?: string
    observed_ttft_ms?: number | null
    steps: ReActStepMetadata[]
  }
  context_and_memory?: {
    models: {
      financial_reasoning_model: string
      coding_and_evaluation_model: string
    }
    turn_context_latency?: TurnContextLatencySummary | null
  }
  agent_trace: AgentTraceStep[]
  agent_identities_used?: AgentIdentitySummary[]
  total_latency_ms: number
}

export interface CachedSectionResponse {
  status: string
  cache_key: string
  cache_hit: boolean
  profile_id: string
  ticker: string
  company_name: string
  fiscal_year: number
  section_id: 'item_1a' | 'item_7'
  token_estimate: number
  sec_filing_url: string
  unstructured_gcs_uri: string
  content: string
  truncated: boolean
  cik?: string
  accession_number?: string
  filing_date?: string
  period_of_report?: string
  gics_sector?: string
  gics_sub_industry?: string
}

export interface SearchScopeFilters {
  selectedTickers: string[]
  gicsSector: string
  sectionId: SectionFilterId
  startYear: number
  endYear: number
  topK: number
}

export interface XbrlFactSelection {
  title: string
  ticker: string
  companyName?: string
  fiscalYear: number
  priorFiscalYear?: number | null
  metricLabel: string
  valueFormatted?: string
  deltaFormatted?: string
  periodOfReport?: string | null
  priorPeriodOfReport?: string | null
  filingDate?: string | null
  accessionNumber?: string | null
  priorAccessionNumber?: string | null
  xbrlConcept?: string | null
  priorXbrlConcept?: string | null
  denominatorXbrlConcept?: string | null
  priorDenominatorXbrlConcept?: string | null
  secFilingUrl?: string | null
  priorSecFilingUrl?: string | null
  derivedFrom?: NumericCitation[]
}

export interface ConversationTurn {
  id: string
  createdAt: string
  userQuery: string
  appliedFilters: SearchScopeFilters
  response: ChatTurnResponse | null
  isLoading: boolean
  progressStage?: string
  progressMessage?: string
  progressMeta?: Record<string, unknown> | null
  progressElapsedMs?: number
  errorMessage?: string
  feedbackRating?: 'up' | 'down'
  feedbackComment?: string
}

export interface AuthenticatedUserProfile {
  sub: string
  email: string
  name: string
  hosted_domain: string | null
  identity_provider: string
  sso_federation: string
  issuer: string
  audience: string
  groups: string[]
  cloud_identity_groups?: string[]
  roles: string[]
  permissions?: string[]
  agent_identity?: string | null
  spiffe_id?: string | null
  issued_at: number
  expires_at: number
  iap_assertion_verified: boolean
}

export interface CloudIdentityRoleSpec {
  role: string
  display_name: string
  description: string
  cloud_identity_groups: string[]
  gcp_iam_roles: string[]
  permissions: string[]
}

export interface AuthDiscoveryConfig {
  status: string
  auth_required: boolean
  access_pattern: string
  identity_provider: {
    name: string
    protocol: string
    google_issuer: string
    iap_issuer: string
    okta_issuer: string | null
    okta_client_id: string | null
    google_client_id: string | null
    authorization_endpoint: string | null
    allowed_workspace_domains: string[]
    required_analyst_groups: string[]
    local_sso_dev_broker_enabled: boolean
  }
  iap: {
    enabled: boolean
    strict_production_mode: boolean
    bearer_header: string
    assertion_header: string
    agent_s2s_assertion_header?: string
    audience_configured: boolean
    session_auth_enabled?: boolean
    default_role?: string | null
  }
  rbac?: {
    management_plane: string
    principal_syntax: string
    cloud_identity_domain: string
    individual_user_bindings_allowed: boolean
    roles: CloudIdentityRoleSpec[]
  }
  service_account?: {
    account_id: string
    email: string
    project_name_email: string
    pattern: string
    project_id: string
    project_name: string
    default_compute_sa_prohibited: boolean
  }
  agent_identities?: Array<{
    agent_name: string
    service_account_id: string
    service_account_email: string
    spiffe_id: string
    allowed_callers: string[]
    allowed_target_services: string[]
    allowed_tools: string[]
    api_permissions: string[]
    gcp_iam_roles: string[]
  }>
}

export interface SsoSessionResponse {
  status: string
  token_type: string
  access_token: string
  expires_at: number
  user: AuthenticatedUserProfile
}
