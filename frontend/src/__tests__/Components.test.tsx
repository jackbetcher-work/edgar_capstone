import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { GroundedMarkdown } from '../components/GroundedMarkdown'
import { HybridScopeBar } from '../components/HybridScopeBar'
import { VarianceTableCard } from '../components/VarianceTableCard'
import { FilingReaderTab } from '../components/inspector/FilingReaderTab'
import type {
  CachedSectionResponse,
  CompanySummary,
  CompanyVarianceReport,
  SearchScopeFilters,
} from '../types/edgar'

afterEach(() => {
  cleanup()
})

const SAMPLE_COMPANIES: CompanySummary[] = [
  {
    ticker: 'NVDA',
    company_name: 'NVIDIA CORP',
    cik: '0001045810',
    gics_sector: 'Information Technology',
    gics_sub_industry: 'Semiconductors',
    latest_fiscal_year: 2026,
    latest_revenue: 215_938_000_000,
    latest_operating_income: 130_387_000_000,
    latest_net_income: 120_067_000_000,
    latest_operating_margin_pct: 60.38,
    latest_net_margin_pct: 55.6,
    sec_filing_url: 'https://www.sec.gov/nvda-2026.htm',
    fiscal_years: [2024, 2025, 2026],
  },
  {
    ticker: 'JPM',
    company_name: 'JPMorgan Chase & Co.',
    cik: '0000019617',
    gics_sector: 'Financials',
    gics_sub_industry: 'Diversified Banks',
    latest_fiscal_year: 2025,
    latest_revenue: 177_556_000_000,
    latest_operating_income: 75_000_000_000,
    latest_net_income: 58_471_000_000,
    latest_operating_margin_pct: 42.24,
    latest_net_margin_pct: 32.93,
    sec_filing_url: 'https://www.sec.gov/jpm-2025.htm',
    fiscal_years: [2024, 2025],
  },
]

const BASE_FILTERS: SearchScopeFilters = {
  selectedTickers: [],
  gicsSector: 'ALL',
  sectionId: 'ALL',
  startYear: 2024,
  endYear: 2026,
  topK: 8,
}

describe('HybridScopeBar', () => {
  it('filters issuers by search, toggles tickers, clamps start/end years, and resets custom scope', () => {
    const onChangeFilters = vi.fn()
    const { rerender } = render(
      <HybridScopeBar
        companies={SAMPLE_COMPANIES}
        sectors={['Information Technology', 'Financials']}
        minYear={2022}
        maxYear={2026}
        filters={BASE_FILTERS}
        onChangeFilters={onChangeFilters}
      />
    )

    // Open popover, filter by sector name, select NVDA
    fireEvent.click(screen.getByRole('button', { name: /EQTY:/i }))
    const searchInput = screen.getByPlaceholderText(/TICKER, ISSUER, OR SECTOR/i)
    fireEvent.change(searchInput, { target: { value: 'financials' } })
    expect(screen.queryByText('NVIDIA CORP')).toBeNull()
    expect(screen.getByText('JPMorgan Chase & Co.')).toBeTruthy()
    fireEvent.click(screen.getByText('JPMorgan Chase & Co.'))
    expect(onChangeFilters).toHaveBeenCalledWith({
      ...BASE_FILTERS,
      selectedTickers: ['JPM'],
    })

    // Year clamping: moving startYear above endYear bumps endYear up
    fireEvent.change(screen.getByLabelText(/FY:/i), { target: { value: '2026' } })
    expect(onChangeFilters).toHaveBeenLastCalledWith({
      ...BASE_FILTERS,
      startYear: 2026,
      endYear: 2026,
    })

    // Moving endYear below startYear pulls startYear down
    fireEvent.change(screen.getByLabelText(/End Fiscal Year/i), { target: { value: '2022' } })
    expect(onChangeFilters).toHaveBeenLastCalledWith({
      ...BASE_FILTERS,
      startYear: 2022,
      endYear: 2022,
    })

    // Active tag quick-remove & Reset button appear when filters are customized
    rerender(
      <HybridScopeBar
        companies={SAMPLE_COMPANIES}
        sectors={['Information Technology', 'Financials']}
        minYear={2022}
        maxYear={2026}
        filters={{ ...BASE_FILTERS, selectedTickers: ['NVDA'], gicsSector: 'Financials' }}
        onChangeFilters={onChangeFilters}
      />
    )
    fireEvent.click(screen.getByRole('button', { name: 'Remove NVDA filter' }))
    expect(onChangeFilters).toHaveBeenLastCalledWith({
      ...BASE_FILTERS,
      selectedTickers: [],
      gicsSector: 'Financials',
    })
    fireEvent.click(screen.getByRole('button', { name: 'RESET' }))
    expect(onChangeFilters).toHaveBeenLastCalledWith({
      selectedTickers: [],
      gicsSector: 'ALL',
      sectionId: 'ALL',
      startYear: 2024,
      endYear: 2026,
      topK: 8,
    })
  })

  it('renders loading and error states inside the issuer popover', () => {
    const { rerender } = render(
      <HybridScopeBar
        companies={[]}
        catalogStatus="loading"
        sectors={[]}
        minYear={2024}
        maxYear={2026}
        filters={BASE_FILTERS}
        onChangeFilters={vi.fn()}
      />
    )
    fireEvent.click(screen.getByRole('button', { name: /EQTY:/i }))
    expect(screen.getByText(/LOADING COMPANIES FROM EDGAR API/i)).toBeTruthy()

    rerender(
      <HybridScopeBar
        companies={[]}
        catalogStatus="error"
        catalogError="HTTP 503 BigQuery down"
        sectors={[]}
        minYear={2024}
        maxYear={2026}
        filters={BASE_FILTERS}
        onChangeFilters={vi.fn()}
      />
    )
    expect(screen.getByText(/HTTP 503 BigQuery down/i)).toBeTruthy()
  })
})

describe('VarianceTableCard', () => {
  function makeReport(
    ticker: string,
    endPeriod: string,
    includeRd: boolean
  ): CompanyVarianceReport {
    return {
      ticker,
      company_name: `${ticker} Corp`,
      cik: '0001045810',
      gics_sector: 'Information Technology',
      gics_sub_industry: 'Semiconductors',
      start_year: 2024,
      end_year: 2026,
      currency: 'USD',
      revenue_variance: {
        metric_name: 'revenue',
        start_year: 2024,
        end_year: 2026,
        start_value: 60_000_000_000,
        end_value: 120_000_000_000,
        absolute_delta: 60_000_000_000,
        percentage_delta: 100.0,
        cagr_pct: 41.42,
        direction: 'increase',
        start_period_of_report: '2024-01-28',
        end_period_of_report: endPeriod,
      },
      operating_income_variance: {
        metric_name: 'operating_income',
        start_year: 2024,
        end_year: 2026,
        start_value: 20_000_000_000,
        end_value: 45_000_000_000,
        absolute_delta: 25_000_000_000,
        percentage_delta: 125.0,
        cagr_pct: 50.0,
        direction: 'increase',
      },
      net_income_variance: {
        metric_name: 'net_income',
        start_year: 2024,
        end_year: 2026,
        start_value: 18_000_000_000,
        end_value: 40_000_000_000,
        absolute_delta: 22_000_000_000,
        percentage_delta: 122.22,
        cagr_pct: 49.07,
        direction: 'increase',
      },
      research_and_development_variance: includeRd
        ? {
            metric_name: 'research_and_development',
            start_year: 2024,
            end_year: 2026,
            start_value: 8_000_000_000,
            end_value: 16_000_000_000,
            absolute_delta: 8_000_000_000,
            percentage_delta: 100.0,
            cagr_pct: 41.42,
            direction: 'increase',
          }
        : {
            metric_name: 'research_and_development',
            start_year: 2024,
            end_year: 2026,
            start_value: null,
            end_value: null,
            absolute_delta: null,
            percentage_delta: null,
            cagr_pct: null,
            direction: 'unavailable',
          },
      operating_margin_bps_delta: 417,
      net_margin_bps_delta: 333,
      annual_series: [
        {
          fiscal_year: 2026,
          prev_fiscal_year: 2025,
          revenue: 120_000_000_000,
          revenue_yoy_delta: 20_000_000_000,
          revenue_yoy_pct: 20.0,
          operating_income: 45_000_000_000,
          operating_income_yoy_delta: 5_000_000_000,
          operating_income_yoy_pct: 12.5,
          net_income: 40_000_000_000,
          net_income_yoy_delta: 4_000_000_000,
          net_income_yoy_pct: 11.1,
          sec_filing_url: 'https://www.sec.gov/example.htm',
        },
      ],
      source_urls: { '2026': 'https://www.sec.gov/example.htm' },
      period_end_dates: { '2024': '2024-01-28', '2026': endPeriod },
    }
  }

  it('shows differing FY-end banner for misaligned peers, expands annual trajectory, and handles empty reports', () => {
    const { container } = render(<VarianceTableCard turnId="t0" reports={[]} />)
    expect(container.firstChild).toBeNull()

    const nvda = makeReport('NVDA', '2026-01-25', true)
    const aapl = makeReport('AAPL', '2026-09-26', false)
    const onInspectSection = vi.fn()

    render(
      <VarianceTableCard
        turnId="t1"
        reports={[nvda, aapl]}
        onInspectFilingSection={onInspectSection}
      />
    )
    expect(screen.getByText('DIFFERING FY-END DATES')).toBeTruthy()
    // Multi-ticker table starts collapsed; expanding reveals annual YoY row for NVDA
    const expandBtn = document.getElementById('btn-expand-annual-t1-NVDA')
    expect(expandBtn).toBeTruthy()
    fireEvent.click(expandBtn!)
    expect(screen.getByText(/\+20\.00%/)).toBeTruthy()
  })
})

describe('GroundedMarkdown', () => {
  it('renders headings, bold/italic/code formatting, warning blockquotes, bullets, and clickable citation + XBRL chips', () => {
    const onSelectCitation = vi.fn()
    const onSelectXbrl = vi.fn()
    const md = [
      '### Executive Summary',
      '#### Key Drivers',
      '> Warning: fiscal calendars differ across peers.',
      '> Standard note with *emphasis* and `us-gaap:Revenues`.',
      '- **Blackwell demand** drove growth [NVDA FY2026 | Item 7. MD&A | NVDA_2026_item_7_0001](https://sec.gov/a)',
      '- Revenue hit $215.94B [NVDA FY2026 | XBRL us-gaap:Revenues | acc:0001045810-26-000021](https://sec.gov/b)',
    ].join('\n')

    const { container } = render(
      <GroundedMarkdown
        turnId="t1"
        markdown={md}
        activeChunkId="NVDA_2026_item_7_0001"
        onSelectCitation={onSelectCitation}
        onSelectXbrlCitation={onSelectXbrl}
      />
    )

    expect(screen.getByRole('heading', { level: 3, name: 'Executive Summary' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 4, name: 'Key Drivers' })).toBeTruthy()
    expect(container.querySelector('strong')?.textContent).toBe('Blackwell demand')
    expect(container.querySelector('em')?.textContent).toBe('emphasis')
    expect(container.querySelector('code')?.textContent).toBe('us-gaap:Revenues')

    // Chips use analyst-readable labels: section topic / financial metric, not raw IDs.
    const citBtn = screen.getByRole('button', { name: 'NVDA FY26 · MD&A' })
    expect(citBtn.className).toContain('citation-chip-active')
    fireEvent.click(citBtn)
    expect(onSelectCitation).toHaveBeenCalledWith(
      expect.objectContaining({ chunkId: 'NVDA_2026_item_7_0001', ticker: 'NVDA', fiscalYear: 2026 })
    )

    const xbrlBtn = screen.getByRole('button', { name: 'NVDA FY26 · Revenue' })
    fireEvent.click(xbrlBtn)
    expect(onSelectXbrl).toHaveBeenCalledWith(
      expect.objectContaining({
        xbrlConcept: 'us-gaap:Revenues',
        accessionNumber: '0001045810-26-000021',
      })
    )
  })
})

describe('FilingReaderTab', () => {
  const cached: CachedSectionResponse = {
    status: 'success',
    cache_key: 'NVDA_2026_item_7',
    profile_id: 'NVDA_2026',
    ticker: 'NVDA',
    company_name: 'NVIDIA CORP',
    fiscal_year: 2026,
    section_id: 'item_7',
    cik: '0001045810',
    accession_number: '0001045810-26-000021',
    period_of_report: '2026-01-25',
    filing_date: '2026-02-25',
    sec_filing_url: 'https://www.sec.gov/nvda-2026.htm',
    unstructured_gcs_uri: 'gs://bucket/unstructured/NVDA/2026_10k.md',
    content:
      '### Overview\n\nData Center revenue surged ($4.5B charge for H20 [export] rules).\n\nSecond paragraph without match.',
    token_estimate: 120,
    truncated: false,
    cache_hit: true,
  }

  it('escapes regex metacharacters in keyword search, highlights active excerpts, clears search, and switches section', () => {
    const onChangeKeyword = vi.fn()
    const onRequest = vi.fn()

    const { container, rerender } = render(
      <FilingReaderTab
        filingTargets={[{ ticker: 'NVDA', fiscalYear: 2026 }]}
        cachedSection={cached}
        isLoadingSection={false}
        sectionError={null}
        highlightedExcerpt={null}
        keyword="($4.5B"
        onChangeKeyword={onChangeKeyword}
        onRequestFilingSection={onRequest}
      />
    )

    // Regex metacharacters `(` `$` `.` match literally inside `<mark class="passage-highlight">`
    const marks = container.querySelectorAll('mark.passage-highlight')
    expect(marks).toHaveLength(1)
    expect(marks[0].textContent).toBe('($4.5B')

    // Clear keyword button triggers onChangeKeyword('')
    fireEvent.click(document.getElementById('btn-clear-filing-keyword')!)
    expect(onChangeKeyword).toHaveBeenCalledWith('')

    // Switch to Item 1A Risk Factors
    fireEvent.click(document.getElementById('btn-doc-section-item_1a')!)
    expect(onRequest).toHaveBeenCalledWith('NVDA', 2026, 'item_1a')

    // Highlighted excerpt renders active highlight mark
    rerender(
      <FilingReaderTab
        filingTargets={[{ ticker: 'NVDA', fiscalYear: 2026 }]}
        cachedSection={cached}
        isLoadingSection={false}
        sectionError={null}
        highlightedExcerpt="Data Center revenue surged"
        keyword=""
        onChangeKeyword={onChangeKeyword}
        onRequestFilingSection={onRequest}
      />
    )
    expect(container.querySelector('mark.passage-highlight-active')?.textContent).toBe(
      'Data Center revenue surged'
    )

    // Loading & error & empty states
    rerender(
      <FilingReaderTab
        filingTargets={[]}
        cachedSection={null}
        isLoadingSection={false}
        sectionError="GCS read failed"
        highlightedExcerpt={null}
        keyword=""
        onChangeKeyword={onChangeKeyword}
        onRequestFilingSection={onRequest}
      />
    )
    expect(screen.getByRole('alert').textContent).toContain('GCS read failed')
  })
})
