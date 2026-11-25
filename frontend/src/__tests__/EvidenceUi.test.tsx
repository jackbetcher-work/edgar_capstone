import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { ChatTurnItem } from '../components/ChatTurnItem'
import { GroundedMarkdown } from '../components/GroundedMarkdown'
import { CitationsTab } from '../components/inspector/CitationsTab'
import type {
  ChatTurnResponse,
  ConversationTurn,
  GroundedCitation,
  MaterialVarianceFlag,
} from '../types/edgar'

const URL_ = 'https://www.sec.gov/Archives/edgar/data/320193/aapl-20240928.htm'

const TABLE_EXCERPT = [
  '[AAPL (Apple Inc.) | FY2024 Form 10-K | Item 7. MD&A — Gross Margin]',
  '',
  '**Gross Margin**',
  '',
  'Products and Services gross margin for 2024, 2023 and 2022 were as follows \\(dollars in millions\\):',
  '',
  '|  | 2024 |  | 2023 |  |',
  '| --- | --- | --- | --- | --- |',
  '| Products | $ | 109,633 | $ | 108,803 |',
  '| Services | 71,050 |  | 60,345 |  |',
].join('\n')

const RISK_EXCERPT =
  'The Company’s operations and performance depend significantly on global and regional economic conditions and adverse economic conditions can materially adversely affect the Company’s business.'

function citation(chunkId: string, overrides: Partial<GroundedCitation> = {}): GroundedCitation {
  return {
    chunk_id: chunkId,
    profile_id: 'AAPL_2024',
    ticker: 'AAPL',
    company_name: 'Apple Inc.',
    gics_sector: 'Information Technology',
    fiscal_year: 2024,
    section_id: 'item_7',
    section_title: 'Item 7. MD&A — Gross Margin',
    excerpt: TABLE_EXCERPT,
    sec_filing_url: URL_,
    unstructured_gcs_uri: '',
    score: 0.8,
    period_of_report: '2024-09-28',
    ...overrides,
  }
}

const GROSS_MARGIN = citation('AAPL_2024_item_7_0005')
const RISK_2024 = citation('AAPL_2024_item_1a_0002', {
  section_id: 'item_1a',
  section_title: 'Item 1A. Risk Factors — Macroeconomic and Industry Risks',
  excerpt: RISK_EXCERPT,
  duplicate_sources: [
    {
      chunk_id: 'AAPL_2023_item_1a_0002',
      ticker: 'AAPL',
      fiscal_year: 2023,
      section_id: 'item_1a',
      section_title: 'Item 1A. Risk Factors — Macroeconomic and Industry Risks',
      sec_filing_url: URL_,
    },
  ],
})

function renderTab(props: Partial<Parameters<typeof CitationsTab>[0]> = {}) {
  const onSelectChunkId = vi.fn()
  const onOpenFilingSection = vi.fn()
  const utils = render(
    <CitationsTab
      citations={[GROSS_MARGIN, RISK_2024]}
      activeChunkId={null}
      sectionFilter="ALL"
      onChangeSectionFilter={vi.fn()}
      onSelectChunkId={onSelectChunkId}
      filingTargets={[]}
      onOpenFilingSection={onOpenFilingSection}
      {...props}
    />
  )
  return { ...utils, onSelectChunkId, onOpenFilingSection }
}

describe('Grounded Chunks panel', () => {
  afterEach(() => cleanup())

  it('shows de-duplicated passages collapsed by default with a readable preview', () => {
    renderTab()
    const toggles = screen.getAllByRole('button', { expanded: false })
    expect(toggles.filter((b) => b.id.startsWith('btn-toggle-source-'))).toHaveLength(2)
    expect(screen.getByText(/1 duplicate merged/)).toBeTruthy()
    expect(screen.getByText('+1 filing')).toBeTruthy()
    // Collapsed: title + preview only, no raw header / chunk id / table body.
    expect(screen.getByText('Gross Margin')).toBeTruthy()
    expect(
      screen.getByText(/^Products and Services gross margin for 2024, 2023 and 2022/)
    ).toBeTruthy()
    expect(screen.queryByRole('table')).toBeNull()
    expect(screen.queryByText(/AAPL_2024_item_7_0005/)).toBeNull()
    expect(screen.queryByText(/Form 10-K \|/)).toBeNull()
  })

  it('expands a passage into structured markdown with a repaired table', () => {
    const { onSelectChunkId, onOpenFilingSection } = renderTab()
    fireEvent.click(document.getElementById('btn-toggle-source-AAPL_2024_item_7_0005')!)
    expect(onSelectChunkId).toHaveBeenCalledWith('AAPL_2024_item_7_0005')

    const table = screen.getByRole('table')
    expect(table.textContent).toContain('$109,633')
    expect(table.textContent).toContain('71,050')
    expect(screen.getByText(/\(dollars in millions\):/)).toBeTruthy()
    expect(screen.getByText(/Period ended Sep 2024/)).toBeTruthy()

    fireEvent.click(screen.getByText('Open in 10-K Section'))
    expect(onOpenFilingSection).toHaveBeenCalledWith(
      'AAPL',
      2024,
      'item_7',
      expect.stringMatching(/^Products and Services gross margin/)
    )
  })

  it('reveals the merged card when an inline citation targets a duplicate, and lets analysts view each filing', () => {
    renderTab({ activeChunkId: 'AAPL_2023_item_1a_0002', focusNonce: 1 })
    const toggle = document.getElementById('btn-toggle-source-AAPL_2024_item_1a_0002')!
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('Appears in:')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'AAPL FY2023' }))
    expect(screen.getByRole('button', { name: 'AAPL FY2023' }).className).toContain('badge-amber')

    // Analyst can collapse it again even while it stays the active chunk.
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
  })

  it('expands and collapses all passages', () => {
    renderTab()
    fireEvent.click(screen.getByRole('button', { name: 'Expand all passages' }))
    expect(screen.getAllByRole('button', { expanded: true }).length).toBe(2)
    fireEvent.click(screen.getByRole('button', { name: 'Collapse all passages' }))
    expect(screen.queryAllByRole('button', { expanded: true })).toHaveLength(0)
  })
})

describe('GroundedMarkdown answer formatting', () => {
  afterEach(() => cleanup())

  it('labels citations by metric / topic and collapses finding evidence quotes', () => {
    const md = [
      '### AAPL margin drivers',
      '#### Key Metrics (deterministic engine)',
      `- **Apple Inc. (AAPL) · Pre-tax income:** $123.5B [AAPL FY2024 | XBRL us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest | acc:0000320193-24-000123](${URL_})`,
      '#### Findings',
      `- Services mix lifted gross margin [AAPL FY2024 | Item 7. MD&A — Gross Margin | AAPL_2024_item_7_0005](${URL_})`,
      '  - *Evidence:* "Services gross margin percentage increased during 2024 compared to 2023 due to a different Services mix."',
    ].join('\n')

    render(
      <GroundedMarkdown
        turnId="t1"
        markdown={md}
        activeChunkId={null}
        onSelectCitation={vi.fn()}
        onSelectXbrlCitation={vi.fn()}
      />
    )

    expect(screen.getByRole('heading', { level: 4, name: 'Key Metrics' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'AAPL FY24 · Pre-tax Income' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'AAPL FY24 · Gross Margin' })).toBeTruthy()
    expect(document.body.textContent).not.toContain('IncomeLossFromContinuingOperations')
    expect(document.body.textContent).not.toContain('AAPL_2024_item_7_0005')

    // Finding stays visible; its verbatim evidence quote is collapsed.
    expect(screen.getByText(/Services mix lifted gross margin/)).toBeTruthy()
    expect(screen.queryByText(/different Services mix/)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Supporting quote/i }))
    expect(screen.getByText(/different Services mix/)).toBeTruthy()
  })
})

describe('ChatTurnItem decluttering', () => {
  afterEach(() => cleanup())

  const FLAG: MaterialVarianceFlag = {
    flag_id: 'AAPL_2024_net_income_yoy',
    ticker: 'AAPL',
    company_name: 'Apple Inc.',
    gics_sector: 'Information Technology',
    fiscal_year: 2024,
    prior_fiscal_year: 2023,
    metric: 'net_income',
    rule: 'yoy_change',
    severity: 'high',
    value: -3.36,
    unit: 'pct',
    threshold: 25,
    prior_value: 97e9,
    current_value: 93.7e9,
    benchmark_value: null,
    peer_count: null,
    description: 'Net income fell.',
    explanation_query: '',
    source: {
      accession_number: '0000320193-24-000123',
      sec_filing_url: URL_,
      period_of_report: '2024-09-28',
      xbrl_concept: 'us-gaap:NetIncomeLoss',
      prior_accession_number: null,
      prior_sec_filing_url: null,
      prior_period_of_report: null,
    },
    explanation: null,
  }

  function turnWith(overrides: Partial<ChatTurnResponse>): ConversationTurn {
    const response: ChatTurnResponse = {
      status: 'success',
      model_armor_blocked: false,
      intent: 'longitudinal_single_company',
      orchestrator: 'RootOrchestrator',
      resolved_scope: { tickers: ['AAPL'], gics_sector: 'ALL', section_id: 'ALL', start_year: 2023, end_year: 2024 },
      answer_markdown: [
        '### Apple net income',
        '> **Warning (Coverage):** AAPL FY2022 10-K is not in the corpus.',
        `Net income declined [AAPL FY2024 | Item 7. MD&A — Gross Margin | AAPL_2024_item_7_0005](${URL_})`,
      ].join('\n'),
      coverage_notices: [
        { code: 'missing_years', severity: 'warning', message: 'AAPL FY2022 10-K is not in the corpus.' },
        { code: 'calendar', severity: 'info', message: 'Apple fiscal year ends in September.' },
      ],
      material_variance_flags: [FLAG],
      material_variance_summary: {
        total_flags: 1,
        returned_flags: 1,
        high_severity_count: 1,
        explained_flags: 0,
        thresholds: {},
      },
      citations: [GROSS_MARGIN, GROSS_MARGIN, RISK_2024],
      fact_check: {
        passed: true,
        checked_metrics_count: 1,
        matched_citations_count: 1,
        has_required_citations: true,
        discrepancies: [],
        model_latency_ms: 3,
      },
      react_execution: { strategy: 'react', iterations: 4 },
      context_and_memory: {
        models: { financial_reasoning_model: 'gemini-pro', coding_and_evaluation_model: 'gemini-flash' },
        turn_context_latency: {
          total_calls: 1,
          total_context_tokens: 10,
          total_cached_context_tokens: 5,
          total_uncached_context_tokens: 5,
          mean_context_latency_ms: 12.3,
          p50_context_latency_ms: 12,
          p95_context_latency_ms: 13,
          mean_total_model_latency_ms: 100,
          p95_ttft_ms: 200,
          ttft_slo_target_ms: 1000,
          ttft_slo_compliance_ratio: 1,
          total_estimated_cache_latency_saved_ms: 0,
        },
      },
      agent_trace: [],
      total_latency_ms: 4200,
      ...overrides,
    }
    return {
      id: 'turn-x',
      createdAt: '2026-11-18T00:00:00Z',
      userQuery: 'Why did Apple net income change?',
      response,
      isLoading: false,
    }
  }

  function renderTurn(turn: ConversationTurn, activeChunkId: string | null = null) {
    return render(
      <ChatTurnItem
        turn={turn}
        activeChunkId={activeChunkId}
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

  it('collapses evidence cards behind a digest header', () => {
    renderTurn(turnWith({}))
    const flagsToggle = screen.getByRole('button', { name: /Material Variance Flags/i })
    expect(flagsToggle.getAttribute('aria-expanded')).toBe('false')
    expect(screen.getByText('1 FLAGGED · 1 HIGH')).toBeTruthy()
    expect(screen.queryByText('-3.36%')).toBeNull()
    fireEvent.click(flagsToggle)
    expect(screen.getByText('-3.36%')).toBeTruthy()
  })

  it('shows each coverage notice once, counts distinct sources, and hides pipeline telemetry', () => {
    renderTurn(turnWith({}))
    expect(screen.getAllByText(/AAPL FY2022 10-K is not in the corpus/)).toHaveLength(1)
    expect(screen.getByText('Apple fiscal year ends in September.')).toBeTruthy()
    expect(screen.getByText('[2 SOURCES]')).toBeTruthy()
    expect(screen.queryByText(/ReAct:/)).toBeNull()
    expect(screen.queryByText(/CTX:/)).toBeNull()
  })

  it('highlights a chip when the active source card is its merged duplicate', () => {
    renderTurn(
      turnWith({
        answer_markdown: `Macro risk persists [AAPL FY2024 | Item 1A. Risk Factors — Macroeconomic and Industry Risks | AAPL_2024_item_1a_0002](${URL_})`,
      }),
      'AAPL_2023_item_1a_0002'
    )
    const chip = screen.getByRole('button', { name: 'AAPL FY24 · Risk Factors' })
    expect(chip.className).toContain('citation-chip-active')
  })
})
