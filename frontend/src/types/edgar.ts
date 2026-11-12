export type SectionFilterId = 'ALL' | 'item_1a' | 'item_7'

export type QueryIntentType =
  | 'longitudinal_single_company'
  | 'multi_company_peer_comparison'
  | 'thematic_trend_tracking'
  | 'chunk_health_monitoring'
  | 'safety_guardrail_block'

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
  latest_operating_margin_pct: number | null
  latest_net_margin_pct: number | null
  sec_filing_url: string
}

export interface CompaniesCatalogResponse {
  status: string
  company_count: number
  total_profiles: number
  min_year: number
  max_year: number
  gics_sectors: string[]
  companies: CompanySummary[]
}

export interface MetricVariance {
  metric_name: 'revenue' | 'operating_income' | 'net_income'
  start_year: number
  end_year: number
  start_value: number | null
  end_value: number | null
  absolute_delta: number | null
  percentage_delta: number | null
  cagr_pct: number | null
  direction: 'increase' | 'decrease' | 'flat' | 'unavailable'
}

export interface AnnualStepVariance {
  fiscal_year: number
  prev_fiscal_year: number | null
  revenue: number | null
  revenue_yoy_delta: number | null
  revenue_yoy_pct: number | null
  operating_income: number | null
  operating_income_yoy_delta: number | null
  operating_income_yoy_pct: number | null
  net_income: number | null
  net_income_yoy_delta: number | null
  net_income_yoy_pct: number | null
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
  currency: string
  revenue_variance: MetricVariance
  operating_income_variance: MetricVariance
  net_income_variance: MetricVariance
  operating_margin_bps_delta: number | null
  net_margin_bps_delta: number | null
  annual_series: AnnualStepVariance[]
  source_urls: Record<string, string>
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
  tool: string
  status: 'completed' | 'warning' | 'blocked'
  duration_ms: number
  summary: string
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
}

export interface ChatTurnResponse {
  status: 'success' | 'blocked' | 'error'
  model_armor_blocked: boolean
  intent: QueryIntentType
  orchestrator: string
  resolved_scope: ResolvedQueryScope
  answer_markdown: string
  variance_reports: CompanyVarianceReport[]
  sector_benchmarks?: SectorBenchmarkRow[]
  citations: GroundedCitation[]
  candidates_searched?: number
  context_cache: ContextCacheInfo | null
  fact_check: FactCheckVerification
  agent_trace: AgentTraceStep[]
  total_latency_ms: number
  connection_mode?: 'live' | 'fallback'
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

export interface ConversationTurn {
  id: string
  createdAt: string
  userQuery: string
  appliedFilters: SearchScopeFilters
  response: ChatTurnResponse | null
  isLoading: boolean
  errorMessage?: string
  feedbackRating?: 'up' | 'down'
  feedbackComment?: string
}
