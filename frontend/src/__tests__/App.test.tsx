import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from '../App'
import { clearStoredBearerToken } from '../services/authService'

// ---------------------------------------------------------------------------
// Backend-shaped fixtures. These mirror the JSON returned by the FastAPI
// endpoints; the frontend itself never synthesizes any of this data.
// ---------------------------------------------------------------------------

const NVDA_FILING_URL =
  'https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm'
const AAPL_FILING_URL =
  'https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm'

const AUTH_CONFIG_OPEN = {
  status: 'ok',
  auth_required: false,
  access_pattern: 'iap_jwt_bearer',
  identity_provider: {
    name: 'Google Workspace (Okta SSO)',
    protocol: 'OIDC / SAML-Federated Workspace + IAP',
    google_issuer: 'https://accounts.google.com',
    iap_issuer: 'https://cloud.google.com/iap',
    okta_issuer: null,
    okta_client_id: null,
    google_client_id: null,
    authorization_endpoint: null,
    allowed_workspace_domains: [],
    required_analyst_groups: [],
    local_sso_dev_broker_enabled: false,
  },
  iap: {
    enabled: false,
    strict_production_mode: false,
    bearer_header: 'Authorization',
    assertion_header: 'X-Goog-IAP-JWT-Assertion',
    audience_configured: false,
  },
}

const COMPANIES_CATALOG = {
  status: 'success',
  company_count: 2,
  total_profiles: 12,
  min_year: 2020,
  max_year: 2026,
  gics_sectors: ['Information Technology'],
  companies: [
    {
      ticker: 'AAPL',
      company_name: 'Apple Inc.',
      cik: '0000320193',
      gics_sector: 'Information Technology',
      gics_sub_industry: 'Technology Hardware, Storage & Peripherals',
      fiscal_years: [2020, 2021, 2022, 2023, 2024, 2025],
      latest_fiscal_year: 2025,
      latest_revenue: 416_161_000_000,
      latest_operating_income: 133_050_000_000,
      latest_net_income: 112_010_000_000,
      latest_operating_margin_pct: 31.97,
      latest_net_margin_pct: 26.92,
      sec_filing_url: AAPL_FILING_URL,
    },
    {
      ticker: 'NVDA',
      company_name: 'NVIDIA CORP',
      cik: '0001045810',
      gics_sector: 'Information Technology',
      gics_sub_industry: 'Semiconductors',
      fiscal_years: [2020, 2021, 2022, 2023, 2024, 2025, 2026],
      latest_fiscal_year: 2026,
      latest_revenue: 215_938_000_000,
      latest_operating_income: 130_387_000_000,
      latest_net_income: 120_067_000_000,
      latest_operating_margin_pct: 60.38,
      latest_net_margin_pct: 55.6,
      sec_filing_url: NVDA_FILING_URL,
    },
  ],
}

const NVDA_CITATION = {
  chunk_id: 'NVDA_2026_item_7_0002',
  profile_id: 'NVDA_2026',
  ticker: 'NVDA',
  company_name: 'NVIDIA CORP',
  gics_sector: 'Information Technology',
  fiscal_year: 2026,
  section_id: 'item_7',
  section_title: 'Item 7. Management’s Discussion and Analysis',
  excerpt: 'Revenue for fiscal year 2026 was $215.9 billion, up 65% from a year ago.',
  sec_filing_url: NVDA_FILING_URL,
  unstructured_gcs_uri: 'gs://edgar-corpus/unstructured/NVDA/2026_10k.md',
  score: 0.912,
}

const CHAT_RESPONSE = {
  status: 'success',
  model_armor_blocked: false,
  intent: 'multi_company_peer_comparison',
  orchestrator: 'RootOrchestrator',
  resolved_scope: {
    tickers: ['NVDA', 'AAPL'],
    gics_sector: 'ALL',
    section_id: 'ALL',
    start_year: 2024,
    end_year: 2026,
  },
  answer_markdown: [
    '### Peer Comparison',
    `- NVIDIA revenue grew **+254.45%** from FY2024 to FY2026 [NVDA FY2026 | XBRL us-gaap:Revenues | acc:0001045810-26-000021](${NVDA_FILING_URL}) [NVDA FY2026 | Item 7 MD&A | NVDA_2026_item_7_0002](${NVDA_FILING_URL})`,
  ].join('\n'),
  coverage_notices: [
    {
      code: 'fiscal_calendar_alignment',
      severity: 'info',
      message: 'Fiscal year-ends differ across peers: NVDA (Jan 2026) vs AAPL (Sep 2025).',
    },
  ],
  sector_coverage: {
    gics_sector: 'Information Technology',
    total_sector_companies: 76,
    companies_analyzed: 76,
    companies_with_matches: 58,
  },
  variance_reports: [
    {
      ticker: 'NVDA',
      company_name: 'NVIDIA CORP',
      cik: '0001045810',
      gics_sector: 'Information Technology',
      gics_sub_industry: 'Semiconductors',
      start_year: 2024,
      end_year: 2026,
      currency: 'USD',
      period_end_dates: { '2024': '2024-01-28', '2026': '2026-01-25' },
      filing_dates: { '2024': '2024-02-21', '2026': '2026-02-25' },
      accession_numbers: {
        '2024': '0001045810-24-000029',
        '2026': '0001045810-26-000021',
      },
      metric_provenance: {
        '2024': { revenue: 'us-gaap:Revenues' },
        '2026': { revenue: 'us-gaap:Revenues' },
      },
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
        start_period_of_report: '2024-01-28',
        end_period_of_report: '2026-01-25',
        start_accession_number: '0001045810-24-000029',
        end_accession_number: '0001045810-26-000021',
        start_xbrl_concept: 'us-gaap:Revenues',
        end_xbrl_concept: 'us-gaap:Revenues',
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
    },
  ],
  citations: [NVDA_CITATION],
  candidates_searched: 48,
  context_cache: null,
  fact_check: {
    passed: true,
    checked_metrics_count: 3,
    matched_citations_count: 1,
    has_required_citations: true,
    discrepancies: [],
    model_latency_ms: 12.3,
  },
  agent_trace: [
    {
      step: 1,
      agent: 'RootOrchestrator',
      tool: 'classify_query_intent',
      status: 'completed',
      duration_ms: 1.1,
      summary: "Routed to 'multi_company_peer_comparison'.",
    },
  ],
  total_latency_ms: 842.5,
}

const SECTION_RESPONSE = {
  status: 'success',
  cache_key: 'NVDA_2026_item_7',
  cache_hit: false,
  profile_id: 'NVDA_2026',
  ticker: 'NVDA',
  company_name: 'NVIDIA CORP',
  fiscal_year: 2026,
  section_id: 'item_7',
  token_estimate: 9120,
  sec_filing_url: NVDA_FILING_URL,
  unstructured_gcs_uri: 'gs://edgar-corpus/unstructured/NVDA/2026_10k.md',
  content:
    '### Fiscal Year 2026 Summary\n\nRevenue for fiscal year 2026 was $215.9 billion, up 65% from a year ago.',
  truncated: false,
  cik: '0001045810',
  accession_number: '0001045810-26-000021',
  filing_date: '2026-02-25',
  period_of_report: '2026-01-25',
  gics_sector: 'Information Technology',
  gics_sub_industry: 'Semiconductors',
}

type RouteOverrides = {
  companies?: () => Promise<Response> | Response
  chat?: () => Promise<Response> | Response
  section?: () => Promise<Response> | Response
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

const sse = (events: Array<{ event: string; data: unknown }>) => {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const item of events) {
        const frame = `event: ${item.event}\ndata: ${JSON.stringify(item.data)}\n\n`
        controller.enqueue(encoder.encode(frame))
      }
      controller.close()
    },
  })
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

function mockBackend(overrides: RouteOverrides = {}) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = typeof input === 'string' ? input : input.toString()
    if (url === '/api/v1/auth/config') return json(AUTH_CONFIG_OPEN)
    if (url === '/api/v1/companies') {
      return overrides.companies ? overrides.companies() : json(COMPANIES_CATALOG)
    }
    if (url === '/api/v1/chat/stream') {
      return overrides.chat
        ? overrides.chat()
        : sse([
            {
              event: 'progress',
              data: {
                stage: 'sector_map',
                message: '28/76 Information Technology companies analyzed (batch 2/6)',
                meta: { companies_analyzed: 28, total_companies: 76 },
                elapsed_ms: 420,
              },
            },
            { event: 'result', data: CHAT_RESPONSE },
          ])
    }
    if (url.startsWith('/api/v1/section')) {
      return overrides.section ? overrides.section() : json(SECTION_RESPONSE)
    }
    if (url === '/api/v1/feedback') return json({ status: 'recorded' })
    return json({ detail: 'Not Found' }, 404)
  })
}

describe('EDGAR <GO> Financial Terminal Workspace', () => {
  beforeEach(() => {
    clearStoredBearerToken()
    vi.restoreAllMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders single h1 heading, terminal dark theme, scope bar, backend coverage monitor, and 10-K pane', async () => {
    mockBackend()
    render(<App />)

    const headings = screen.getAllByRole('heading', { level: 1 })
    expect(headings).toHaveLength(1)
    expect(headings[0].textContent).toBe('EDGAR Analyst')

    expect(document.documentElement.getAttribute('data-theme')).toBe('terminal')

    expect(
      await screen.findByText(/Query audited financials and grounded 10-K disclosures/i)
    ).toBeTruthy()

    expect(screen.getByLabelText(/GICS Sector/i)).toBeTruthy()
    expect(screen.getByLabelText(/Form 10-K Section/i)).toBeTruthy()
    expect(screen.getByText(/Grounded Chunks/i)).toBeTruthy()

    // Coverage monitor reflects exactly what /api/v1/companies returned.
    expect(
      await screen.findByText(/S&P 500 COVERAGE MONITOR \(2 EQUITIES\)/i)
    ).toBeTruthy()
    expect(screen.getByText('2 ISSUERS · 12 FILINGS')).toBeTruthy()
    expect(screen.getByText('READY')).toBeTruthy()
  })

  it('executes a peer comparison RAG query, renders backend variance math, inline citations, and opens 10-K section', async () => {
    const fetchSpy = mockBackend()
    render(<App />)

    const starterBtn = await screen.findByText(
      /NVIDIA vs\. Apple: FY2024–FY2026 Revenue, Margin & AI Drivers/i
    )
    fireEvent.click(starterBtn)

    await waitFor(() => {
      expect(screen.getByText(/Fact-Check Verified/i)).toBeTruthy()
    })

    expect(
      fetchSpy.mock.calls.some(([input]) => String(input) === '/api/v1/chat/stream')
    ).toBe(true)

    // Coverage notice & sector coverage pill rendered
    expect(screen.getByText(/Fiscal year-ends differ across peers/i)).toBeTruthy()
    expect(screen.getByText(/76\/76 Information Technology ISSUERS/i)).toBeTruthy()

    // Inline XBRL citation chip opens XBRL Fact Provenance tab
    const xbrlChip = await screen.findByTitle(/Inspect XBRL fact NVDA FY2026/i)
    fireEvent.click(xbrlChip)
    expect(await screen.findByText(/SEC XBRL NUMERIC PROVENANCE/i)).toBeTruthy()
    expect(screen.getByText('0001045810-26-000021')).toBeTruthy()

    // Deterministic Variance Table rendered from the backend payload, collapsed
    // beneath the answer with a digest; expanding reveals period-end dates.
    expect(screen.getByText('Financial Variances')).toBeTruthy()
    expect(screen.getByText(/NVDA · FY24→FY26 · XBRL · USD/)).toBeTruthy()
    expect(screen.queryByText(/Jan 2024 → Jan 2026/i)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Financial Variances/i }))
    expect(screen.getByText(/Jan 2024 → Jan 2026/i)).toBeTruthy()
    expect(screen.getAllByText(/\+254\.45%/i).length).toBeGreaterThan(0)

    // Inline text citation chip rendered and interactive
    const inlineChips = await screen.findAllByTitle(/Inspect NVDA FY2026/i)
    expect(inlineChips.length).toBeGreaterThan(0)
    fireEvent.click(inlineChips[0])

    // Open in Full 10-K Section reader
    const openSectionButtons = await screen.findAllByText(/Open in 10-K Section/i)
    fireEvent.click(openSectionButtons[0])

    await waitFor(() => {
      expect(screen.getByText(/UNITED STATES SEC · FORM 10-K/i)).toBeTruthy()
    })
    expect(screen.getByText(/ACC: 0001045810-26-000021/)).toBeTruthy()
  })

  it('focuses the EDGR> command line via / shortcut and executes a query with <GO>', async () => {
    mockBackend()
    render(<App />)
    await screen.findByText(/EDGR <GO>/i)

    fireEvent.keyDown(window, { key: '/' })
    const cmdInput = screen.getByLabelText(
      /Ask a natural language financial or SEC 10-K question/i
    )
    expect(document.activeElement).toBe(cmdInput)

    fireEvent.change(cmdInput, {
      target: { value: 'NVDA FY2024 to FY2026 revenue variance' },
    })
    fireEvent.click(screen.getByRole('button', { name: /^<GO>$/ }))

    await waitFor(() => {
      expect(screen.getByText(/CMD>/i)).toBeTruthy()
      expect(
        screen.getByText('NVDA FY2024 to FY2026 revenue variance')
      ).toBeTruthy()
    })
  })

  it('renders a backend-unreachable error and no fabricated answer when the chat request fails', async () => {
    mockBackend({
      chat: () => Promise.reject(new TypeError('Failed to fetch')),
    })
    render(<App />)

    const starterBtn = await screen.findByText(
      /NVIDIA vs\. Apple: FY2024–FY2026 Revenue, Margin & AI Drivers/i
    )
    fireEvent.click(starterBtn)

    const alert = await screen.findByText(/Backend unreachable/i)
    expect(alert.textContent).toContain('could not reach the EDGAR API')
    expect(alert.textContent).toContain('Failed to fetch')
    expect(alert.textContent).toContain('No data was shown')

    // Nothing synthesized: no answer, variance table, fact-check badge, or citations.
    expect(screen.queryByText(/Fact-Check Verified/i)).toBeNull()
    expect(screen.queryByText(/Financial Variances/i)).toBeNull()
    expect(screen.queryByText(/\+254\.45%/)).toBeNull()
    expect(screen.queryAllByTitle(/Inspect NVDA FY2026/i)).toHaveLength(0)
    expect(screen.getByText(/NO 10-K PASSAGES LOADED/i)).toBeTruthy()
  })

  it('surfaces the HTTP status and backend detail when the chat endpoint returns an error', async () => {
    mockBackend({
      chat: () => json({ detail: 'Vertex AI quota exhausted' }, 503),
    })
    render(<App />)

    fireEvent.click(
      await screen.findByText(
        /NVIDIA vs\. Apple: FY2024–FY2026 Revenue, Margin & AI Drivers/i
      )
    )

    const alert = await screen.findByText(/HTTP 503/)
    expect(alert.textContent).toContain('Vertex AI quota exhausted')
    expect(alert.textContent).toContain('No data was shown')
    expect(screen.queryByText(/Fact-Check Verified/i)).toBeNull()
  })

  it('shows an error state instead of a company list when /api/v1/companies is unreachable', async () => {
    mockBackend({
      companies: () => Promise.reject(new TypeError('Failed to fetch')),
    })
    render(<App />)

    const catalogAlert = await screen.findByText(/COVERAGE CATALOG UNAVAILABLE/i)
    expect(catalogAlert.textContent).toMatch(/Backend unreachable/i)
    expect(screen.getByText('BACKEND OFFLINE')).toBeTruthy()
    expect(screen.getByText('— ISSUERS · — FILINGS')).toBeTruthy()
    expect(screen.queryByText(/COVERAGE MONITOR/i)).toBeNull()
    expect(screen.queryByText('NVIDIA CORP')).toBeNull()

    // Scope bar issuer picker shows the error rather than invented issuers.
    fireEvent.click(screen.getByRole('button', { name: /EQTY:/i }))
    expect(await screen.findByText(/COMPANY CATALOG UNAVAILABLE/i)).toBeTruthy()
    expect(document.querySelector('[id^="issuer-option-"]')).toBeNull()
  })

  it('shows an error in the 10-K reader instead of synthesized section text when /api/v1/section fails', async () => {
    mockBackend({
      section: () => json({ detail: 'Section not found in corpus' }, 404),
    })
    render(<App />)

    const tenKButtons = await screen.findAllByRole('button', { name: '[10-K]' })
    fireEvent.click(tenKButtons[0])

    const sectionAlert = await screen.findByText(/Section not found in corpus/)
    expect(sectionAlert.textContent).toContain('HTTP 404')
    expect(screen.queryByText(/UNITED STATES SEC · FORM 10-K/i)).toBeNull()
  })

  it('renders clarification card with clickable follow-up chips and proper projection block badge', async () => {
    let callCount = 0
    mockBackend({
      chat: () => {
        callCount += 1
        if (callCount === 1) {
          return sse([
            {
              event: 'result',
              data: {
                ...CHAT_RESPONSE,
                status: 'clarification',
                intent: 'clarification_needed',
                answer_markdown:
                  '### Clarification Needed\n\nPlease specify which S&P 500 company or GICS sector you want to analyze.',
                variance_reports: [],
                citations: [],
                suggested_followups: [
                  'Compare NVDA vs AAPL revenue FY2024 to FY2026',
                ],
                coverage_notices: [],
                sector_coverage: null,
              },
            },
          ])
        }
        return sse([
          {
            event: 'result',
            data: {
              ...CHAT_RESPONSE,
              status: 'blocked',
              model_armor_blocked: true,
              intent: 'projection_compliance_block',
              answer_markdown:
                '### Forecast Blocked\n\nHistorical audited SEC 10-K analysis only.',
              variance_reports: [],
              citations: [],
              suggested_followups: [],
              coverage_notices: [],
              sector_coverage: null,
            },
          },
        ])
      },
    })
    render(<App />)

    const cmdInput = await screen.findByLabelText(
      /Ask a natural language financial or SEC 10-K question/i
    )
    fireEvent.change(cmdInput, { target: { value: 'How did revenue change?' } })
    fireEvent.click(screen.getByRole('button', { name: /^<GO>$/ }))

    expect(await screen.findByText('CLARIFY')).toBeTruthy()
    const chip = await screen.findByRole('button', {
      name: 'Compare NVDA vs AAPL revenue FY2024 to FY2026',
    })
    fireEvent.click(chip)

    expect(await screen.findByText('PROJECTION BLOCK')).toBeTruthy()
    expect(screen.getByText(/Projection Compliance Guardrail/i)).toBeTruthy()
  })

  it('handles null-intent scope_resolved and agent_tool SSE stages, rendering a qualitative answer without deterministic cards', async () => {
    const RISK_CITATION = {
      ...NVDA_CITATION,
      chunk_id: 'NVDA_2026_item_1a_0003',
      section_id: 'item_1a',
      section_title: 'Item 1A. Risk Factors',
      excerpt: 'Export controls could adversely affect our data center business.',
    }
    mockBackend({
      chat: () =>
        sse([
          {
            event: 'progress',
            data: {
              stage: 'scope_resolved',
              message: 'Scope resolved: NVDA FY2024–FY2026',
              meta: { intent: null, tickers: ['NVDA'] },
              elapsed_ms: 40,
            },
          },
          {
            event: 'progress',
            data: {
              stage: 'agent_tool',
              message: 'Agent invoked retrieve_10k_passages',
              meta: { tool: 'retrieve_10k_passages' },
              elapsed_ms: 120,
            },
          },
          {
            event: 'result',
            data: {
              ...CHAT_RESPONSE,
              intent: 'longitudinal_single_company',
              intent_source: 'agent',
              answer_markdown: `### Export-Control Risk\n\nNVIDIA added export-control risk language [NVDA FY2026 | Item 1A Risk Factors | NVDA_2026_item_1a_0003](${NVDA_FILING_URL}) while revenue reached $215.9B [NVDA FY2026 | XBRL us-gaap:Revenues | acc:0001045810-26-000021](${NVDA_FILING_URL})`,
              coverage_notices: [],
              sector_coverage: null,
              variance_reports: [],
              material_variance_flags: [],
              material_variance_summary: null,
              disclosure_deltas: null,
              citations: [RISK_CITATION],
            },
          },
        ]),
    })
    render(<App />)

    const cmdInput = await screen.findByLabelText(
      /Ask a natural language financial or SEC 10-K question/i
    )
    fireEvent.change(cmdInput, {
      target: { value: 'What export-control risks did NVIDIA disclose?' },
    })
    fireEvent.click(screen.getByRole('button', { name: /^<GO>$/ }))

    expect(await screen.findByText('Export-Control Risk')).toBeTruthy()
    expect(screen.getByText(/Fact-Check Verified/i)).toBeTruthy()
    expect(screen.queryByText(/Financial Variances/i)).toBeNull()
    expect(screen.queryByLabelText('Material Variance Flags')).toBeNull()
    expect(screen.queryByText(/Disclosure Changes YoY/i)).toBeNull()
    expect(document.querySelector('[id^="deterministic-evidence-"]')).toBeNull()

    // Inline XBRL chip with no variance report still opens the fact inspector
    // (variance_reports lookup gracefully no-ops when the tool was not called).
    fireEvent.click(await screen.findByTitle(/Inspect XBRL fact NVDA FY2026/i))
    expect(await screen.findByText(/SEC XBRL NUMERIC PROVENANCE/i)).toBeTruthy()
    expect(screen.getByText('0001045810-26-000021')).toBeTruthy()
  })
})
