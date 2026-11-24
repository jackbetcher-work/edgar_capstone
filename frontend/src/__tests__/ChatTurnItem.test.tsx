import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { ChatTurnItem } from '../components/ChatTurnItem'
import type {
  ChatTurnResponse,
  CompanyVarianceReport,
  ConversationTurn,
  DisclosureDeltaResult,
  GroundedCitation,
  MaterialVarianceFlag,
} from '../types/edgar'

// ---------------------------------------------------------------------------
// Backend-shaped fixtures for the agent-invoked-tools contract: deterministic
// payloads are empty / null unless the agent actually called the engine tool.
// ---------------------------------------------------------------------------

const NVDA_FILING_URL =
  'https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm'

const VARIANCE_CARD_LABEL = 'Deterministic Financial Variance Engine Output'
const FLAGS_CARD_LABEL = 'Material Variance Flags'
const DELTA_CARD_LABEL = 'Year-over-Year Disclosure Deltas'

function citation(
  chunkId: string,
  fy: number,
  section: 'item_1a' | 'item_7'
): GroundedCitation {
  return {
    chunk_id: chunkId,
    profile_id: `NVDA_${fy}`,
    ticker: 'NVDA',
    company_name: 'NVIDIA CORP',
    gics_sector: 'Information Technology',
    fiscal_year: fy,
    section_id: section,
    section_title: section === 'item_1a' ? 'Item 1A. Risk Factors' : 'Item 7. MD&A',
    excerpt: 'excerpt',
    sec_filing_url: NVDA_FILING_URL,
    unstructured_gcs_uri: '',
    score: 0.9,
  }
}

const VARIANCE_REPORT: CompanyVarianceReport = {
  ticker: 'NVDA',
  company_name: 'NVIDIA CORP',
  cik: '0001045810',
  gics_sector: 'Information Technology',
  gics_sub_industry: 'Semiconductors',
  start_year: 2024,
  end_year: 2026,
  currency: 'USD',
  period_end_dates: { '2024': '2024-01-28', '2026': '2026-01-25' },
  revenue_variance: {
    metric_name: 'revenue',
    start_year: 2024,
    end_year: 2026,
    start_value: 60_922_000_000,
    end_value: 215_938_000_000,
    absolute_delta: 155_016_000_000,
    percentage_delta: 254.45,
    cagr_pct: 88.27,
    direction: 'increase',
  },
  operating_income_variance: {
    metric_name: 'operating_income',
    start_year: 2024,
    end_year: 2026,
    start_value: 32_972_000_000,
    end_value: 130_387_000_000,
    absolute_delta: 97_415_000_000,
    percentage_delta: 295.45,
    cagr_pct: 98.86,
    direction: 'increase',
  },
  net_income_variance: {
    metric_name: 'net_income',
    start_year: 2024,
    end_year: 2026,
    start_value: 29_760_000_000,
    end_value: 120_067_000_000,
    absolute_delta: 90_307_000_000,
    percentage_delta: 303.45,
    cagr_pct: 100.86,
    direction: 'increase',
  },
  operating_margin_bps_delta: 626.0,
  net_margin_bps_delta: 675.0,
  annual_series: [],
  source_urls: { '2026': NVDA_FILING_URL },
}

const MATERIAL_FLAG: MaterialVarianceFlag = {
  flag_id: 'NVDA_2025_revenue_yoy_change',
  ticker: 'NVDA',
  company_name: 'NVIDIA CORP',
  gics_sector: 'Information Technology',
  fiscal_year: 2025,
  prior_fiscal_year: 2024,
  metric: 'revenue',
  rule: 'yoy_change',
  severity: 'high',
  value: 114.2,
  unit: 'pct',
  threshold: 25,
  prior_value: 60.9e9,
  current_value: 130.5e9,
  benchmark_value: null,
  peer_count: null,
  description: 'Revenue rose +114.20% YoY in FY2025.',
  explanation_query: '',
  source: {
    accession_number: '0001045810-25-000023',
    sec_filing_url: NVDA_FILING_URL,
    period_of_report: '2025-01-26',
    xbrl_concept: 'us-gaap:Revenues',
    prior_accession_number: null,
    prior_sec_filing_url: null,
    prior_period_of_report: null,
  },
  explanation: null,
}

const SUCCESS_DELTAS: DisclosureDeltaResult = {
  status: 'success',
  theme: 'AI risk',
  summary: {
    added: 1,
    removed: 0,
    modified: 0,
    unchanged: 0,
    year_pairs: 1,
    companies_analyzed: 1,
  },
  companies: [
    {
      ticker: 'NVDA',
      company_name: 'NVIDIA CORP',
      gics_sector: 'Information Technology',
      fiscal_years_available: [2024, 2025],
      missing_fiscal_years: [],
      theme_mentions: { '2024': 47, '2025': 64 },
      year_pairs: [
        {
          prior_fiscal_year: 2024,
          current_fiscal_year: 2025,
          counts: { added: 1, modified: 0, removed: 0, unchanged: 0 },
          deltas: [],
        },
      ],
    },
  ],
}

function buildResponse(overrides: Partial<ChatTurnResponse> = {}): ChatTurnResponse {
  return {
    status: 'success',
    model_armor_blocked: false,
    intent: 'longitudinal_single_company',
    intent_source: 'agent',
    orchestrator: 'RootOrchestrator',
    resolved_scope: {
      tickers: ['NVDA'],
      gics_sector: 'ALL',
      section_id: 'item_1a',
      start_year: 2024,
      end_year: 2026,
    },
    answer_markdown: [
      '### Direct Answer',
      `NVIDIA expanded its export-control risk disclosures [NVDA FY2026 | Item 1A Risk Factors | NVDA_2026_item_1a_0001](${NVDA_FILING_URL})`,
    ].join('\n'),
    coverage_notices: [],
    sector_coverage: null,
    variance_reports: [],
    material_variance_flags: [],
    material_variance_summary: null,
    disclosure_deltas: null,
    citations: [citation('NVDA_2026_item_1a_0001', 2026, 'item_1a')],
    fact_check: {
      passed: true,
      checked_metrics_count: 0,
      matched_citations_count: 1,
      has_required_citations: true,
      discrepancies: [],
      model_latency_ms: 4.2,
    },
    agent_trace: [],
    total_latency_ms: 512,
    ...overrides,
  }
}

function buildTurn(response: ChatTurnResponse | null, extra: Partial<ConversationTurn> = {}): ConversationTurn {
  return {
    id: 'turn-test',
    createdAt: '2026-11-18T00:00:00Z',
    userQuery: 'How did NVIDIA risk factors change?',
    response,
    isLoading: false,
    ...extra,
  }
}

function renderTurn(turn: ConversationTurn) {
  return render(
    <ChatTurnItem
      turn={turn}
      activeChunkId={null}
      onSelectInlineCitation={vi.fn()}
      onSelectXbrlCitation={vi.fn()}
      onInspectXbrlFact={vi.fn()}
      onSelectFollowup={vi.fn()}
      onFocusTurnCitations={vi.fn()}
      onInspectFilingSection={vi.fn()}
      onSubmitFeedback={vi.fn()}
    />
  )
}

const isAfter = (later: Element, earlier: Element) =>
  Boolean(earlier.compareDocumentPosition(later) & Node.DOCUMENT_POSITION_FOLLOWING)

describe('ChatTurnItem — agent-invoked deterministic evidence', () => {
  afterEach(() => cleanup())

  it('qualitative risk-factor turn renders the narrative only, with no deterministic cards', () => {
    renderTurn(buildTurn(buildResponse()))

    expect(screen.getByRole('region', { name: 'Grounded Answer' })).toBeTruthy()
    expect(screen.getByText('Direct Answer')).toBeTruthy()
    expect(screen.getAllByTitle(/Inspect NVDA FY2026/i).length).toBeGreaterThan(0)

    expect(screen.queryByLabelText(VARIANCE_CARD_LABEL)).toBeNull()
    expect(screen.queryByText(/Financial Variances/i)).toBeNull()
    expect(screen.queryByLabelText(FLAGS_CARD_LABEL)).toBeNull()
    expect(screen.queryByLabelText(DELTA_CARD_LABEL)).toBeNull()
    expect(screen.queryByText(/Disclosure Changes YoY/i)).toBeNull()
    expect(document.getElementById('deterministic-evidence-turn-test')).toBeNull()
  })

  it('tolerates omitted deterministic fields (undefined variance_reports / flags / deltas)', () => {
    const resp = buildResponse()
    delete resp.variance_reports
    delete resp.material_variance_flags
    delete resp.material_variance_summary
    delete resp.disclosure_deltas
    delete resp.sector_coverage
    renderTurn(buildTurn({ ...resp, citations: [] }))

    expect(screen.getByText('Direct Answer')).toBeTruthy()
    expect(document.getElementById('deterministic-evidence-turn-test')).toBeNull()
    expect(document.getElementById('btn-focus-citations-turn-test')).toBeNull()
  })

  it('quantitative turn renders the VarianceTableCard after the narrative', () => {
    renderTurn(
      buildTurn(
        buildResponse({
          intent: 'longitudinal_single_company',
          answer_markdown: '### Revenue Variance\n\nNVIDIA revenue grew +254.45% FY2024→FY2026.',
          variance_reports: [VARIANCE_REPORT],
        })
      )
    )

    const narrative = screen.getByRole('region', { name: 'Grounded Answer' })
    const varianceCard = screen.getByLabelText(VARIANCE_CARD_LABEL)
    expect(narrative.textContent).toContain('Revenue Variance')
    expect(isAfter(varianceCard, narrative)).toBe(true)

    const evidence = document.getElementById('deterministic-evidence-turn-test')
    expect(evidence).toBeTruthy()
    expect(evidence?.contains(varianceCard)).toBe(true)
    expect(isAfter(evidence as Element, narrative)).toBe(true)

    expect(screen.queryByLabelText(FLAGS_CARD_LABEL)).toBeNull()
    expect(screen.queryByLabelText(DELTA_CARD_LABEL)).toBeNull()
  })

  it('anomaly turn with flags renders the MaterialFlagsCard below the narrative', () => {
    renderTurn(
      buildTurn(
        buildResponse({
          answer_markdown: '### Material Variances\n\nOne high-severity revenue swing was flagged.',
          material_variance_flags: [MATERIAL_FLAG],
          material_variance_summary: {
            total_flags: 3,
            returned_flags: 1,
            high_severity_count: 1,
            explained_flags: 0,
            thresholds: { yoy_change_pct: 25 },
          },
        })
      )
    )

    const narrative = screen.getByRole('region', { name: 'Grounded Answer' })
    const flagsCard = screen.getByLabelText(FLAGS_CARD_LABEL)
    expect(isAfter(flagsCard, narrative)).toBe(true)
    expect(screen.getByText(/1 OF 3/)).toBeTruthy()
    expect(screen.queryByLabelText(VARIANCE_CARD_LABEL)).toBeNull()
    expect(screen.queryByLabelText(DELTA_CARD_LABEL)).toBeNull()
  })

  it('thematic turn with a successful disclosure_deltas payload renders the DisclosureDeltaCard', () => {
    renderTurn(
      buildTurn(
        buildResponse({
          intent: 'thematic_trend_tracking',
          disclosure_deltas: SUCCESS_DELTAS,
        })
      )
    )

    const narrative = screen.getByRole('region', { name: 'Grounded Answer' })
    const deltaCard = screen.getByLabelText(DELTA_CARD_LABEL)
    expect(screen.getByText('Disclosure Changes YoY')).toBeTruthy()
    expect(isAfter(deltaCard, narrative)).toBe(true)
    expect(document.getElementById('deterministic-evidence-turn-test')).toBeTruthy()
  })

  it.each<[string, DisclosureDeltaResult]>([
    ['status error', { status: 'error', message: 'No filings matched the theme.' }],
    ['zero companies', { ...SUCCESS_DELTAS, companies: [] }],
    ['missing companies', { status: 'success', theme: 'AI risk' }],
  ])('does not render the DisclosureDeltaCard for a %s payload', (_label, deltas) => {
    renderTurn(
      buildTurn(
        buildResponse({
          intent: 'thematic_trend_tracking',
          disclosure_deltas: deltas,
        })
      )
    )

    expect(screen.getByText('Direct Answer')).toBeTruthy()
    expect(screen.queryByLabelText(DELTA_CARD_LABEL)).toBeNull()
    expect(screen.queryByText('Disclosure Changes YoY')).toBeNull()
    expect(document.getElementById('deterministic-evidence-turn-test')).toBeNull()
  })

  it('shows a tool-agnostic loading fallback while the agent is working', () => {
    renderTurn(buildTurn(null, { isLoading: true }))
    expect(screen.getByText('ANALYZING QUERY & RETRIEVING 10-K EVIDENCE...')).toBeTruthy()
    expect(screen.queryByText(/COMPUTING XBRL VARIANCES/i)).toBeNull()
  })

  it('shows agent_tool progress messages from the SSE stream', () => {
    renderTurn(
      buildTurn(null, {
        isLoading: true,
        progressStage: 'agent_tool',
        progressMessage: 'Agent invoked calculate_period_variances',
        progressMeta: { tool: 'calculate_period_variances' },
      })
    )
    expect(screen.getByText('Agent invoked calculate_period_variances')).toBeTruthy()
  })

  it('renders live sector_map_progress percentage and canonical sector_coverage badge', () => {
    const { rerender } = render(
      <ChatTurnItem
        turn={buildTurn(null, {
          isLoading: true,
          progressStage: 'sector_map_progress',
          progressMessage: '38/76 Information Technology companies analyzed (batch 3/6)',
          progressMeta: {
            analyzed: 38,
            total: 76,
            issuers_in_sector: 76,
            batch_completed: 3,
            total_batches: 6,
          },
        })}
        activeChunkId={null}
        onSelectInlineCitation={vi.fn()}
        onFocusTurnCitations={vi.fn()}
        onInspectFilingSection={vi.fn()}
        onSubmitFeedback={vi.fn()}
      />
    )
    expect(screen.getByText('50%')).toBeTruthy()

    rerender(
      <ChatTurnItem
        turn={buildTurn(
          buildResponse({
            sector_coverage: {
              gics_sector: 'Information Technology',
              issuers_in_sector: 76,
              issuers_analyzed: 74,
              issuers_with_theme_matches: 58,
              batches_processed: 6,
            },
          })
        )}
        activeChunkId={null}
        onSelectInlineCitation={vi.fn()}
        onFocusTurnCitations={vi.fn()}
        onInspectFilingSection={vi.fn()}
        onSubmitFeedback={vi.fn()}
      />
    )
    const badge = screen.getByText(/74\/76 Information Technology ISSUERS · 58 MATCHED/i)
    expect(badge).toBeTruthy()
    expect(badge.getAttribute('title')).toContain('6 parallel batches')
  })

  it('renders verified structured_report JSON directly with XBRL chips, citation chips, and collapsible quotes', () => {
    const onSelectInlineCitation = vi.fn()
    const onInspectXbrlFact = vi.fn()
    const cit = citation('NVDA_2026_item_7_0001', 2026, 'item_7')
    render(
      <ChatTurnItem
        turn={buildTurn(
          buildResponse({
            answer_markdown: 'Fallback markdown that should NOT be rendered when structured_report is present.',
            variance_reports: [VARIANCE_REPORT],
            citations: [cit],
            structured_report: {
              intent: 'longitudinal_single_company',
              headline: 'NVIDIA revenue surged on Blackwell data center demand.',
              key_metrics: [{ ticker: 'NVDA', metric: 'revenue' }],
              findings: [
                {
                  statement: 'Data center compute shipments drove FY2026 top-line expansion.',
                  ticker: 'NVDA',
                  fiscal_year: 2026,
                  chunk_ids: [cit.chunk_id],
                  evidence_quote: 'Data center demand drove revenue growth.',
                },
              ],
            },
          })
        )}
        activeChunkId={null}
        onSelectInlineCitation={onSelectInlineCitation}
        onInspectXbrlFact={onInspectXbrlFact}
        onFocusTurnCitations={vi.fn()}
        onInspectFilingSection={vi.fn()}
        onSubmitFeedback={vi.fn()}
      />
    )

    expect(
      screen.getByText('NVIDIA revenue surged on Blackwell data center demand.')
    ).toBeTruthy()
    expect(screen.queryByText(/Fallback markdown/)).toBeNull()
    expect(
      screen.getByText('Data center compute shipments drove FY2026 top-line expansion.')
    ).toBeTruthy()

    // Clicking the deterministic XBRL chip inspects the XBRL fact
    fireEvent.click(screen.getByRole('button', { name: /NVDA FY26 · Revenue/i }))
    expect(onInspectXbrlFact).toHaveBeenCalledTimes(1)

    // Clicking the finding citation chip focuses the cited 10-K chunk
    fireEvent.click(screen.getByRole('button', { name: /NVDA FY26 · MD&A/i }))
    expect(onSelectInlineCitation).toHaveBeenCalledTimes(1)
    expect(onSelectInlineCitation.mock.calls[0][0].chunkId).toBe(cit.chunk_id)

    // Evidence quote starts collapsed and expands on click
    expect(screen.queryByText(/Data center demand drove revenue growth/)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Supporting quote/i }))
    expect(screen.getByText(/Data center demand drove revenue growth/)).toBeTruthy()
  })
})
