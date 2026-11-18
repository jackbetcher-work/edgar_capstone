import { describe, expect, it } from 'vitest'
import type {
  AnnualStepVariance,
  CompanyVarianceReport,
  MaterialVarianceFlag,
  MetricVariance,
} from '../types/edgar'
import {
  clean10kDocumentBlock,
  clean10kExcerpt,
  formatPeriodEndShort,
  formatSignedBps,
  formatSignedDeltaUsd,
  formatSignedPercent,
  formatUsdCompact,
  getIntentDisplayMeta,
  splitTextByInlineCitations,
} from '../utils/formatters'
import {
  buildXbrlFactFromAnnualStep,
  buildXbrlFactFromFlag,
  buildXbrlFactFromInlineToken,
  buildXbrlFactFromVariance,
} from '../utils/xbrlFactSelection'

describe('formatters utilities', () => {
  it('formats USD compact values, signed deltas, percents, and bps across null/NaN/negatives', () => {
    expect(formatUsdCompact(null)).toBe('—')
    expect(formatUsdCompact(undefined)).toBe('—')
    expect(formatUsdCompact(Number.NaN)).toBe('—')
    expect(formatUsdCompact(215_938_000_000)).toBe('$215.94B')
    expect(formatUsdCompact(-5_200_000_000)).toBe('-$5.20B')
    expect(formatUsdCompact(42_500_000)).toBe('$42.50M')
    expect(formatUsdCompact(9500)).toBe('$9,500')

    expect(formatSignedDeltaUsd(150_000_000_000)).toBe('+$150.00B')
    expect(formatSignedDeltaUsd(-4_500_000_000)).toBe('-$4.50B')
    expect(formatSignedDeltaUsd(0)).toBe('$0')
    expect(formatSignedDeltaUsd(null)).toBe('—')

    expect(formatSignedPercent(254.45)).toBe('+254.45%')
    expect(formatSignedPercent(-12.3)).toBe('-12.30%')
    expect(formatSignedPercent(0)).toBe('0.00%')
    expect(formatSignedPercent(undefined)).toBe('—')

    expect(formatSignedBps(626.4)).toBe('+626 bps')
    expect(formatSignedBps(-571)).toBe('-571 bps')
    expect(formatSignedBps(null)).toBe('—')
  })

  it('cleans escaped SEC 10-K punctuation and formats short period-end dates', () => {
    expect(clean10kExcerpt('Item\\ 7\\.\\n\\nRevenue   \\(FY2026\\)')).toBe('Item\\ 7.\\n\\nRevenue (FY2026)')
    expect(clean10kExcerpt('Item 7\\.   Revenue\n\n\\(USD\\)  \\-  high')).toBe('Item 7. Revenue (USD) - high')
    expect(clean10kDocumentBlock('  Line 1\\.\nLine 2\\-end  ')).toBe('Line 1.\nLine 2-end')

    expect(formatPeriodEndShort('2026-01-25')).toBe('Jan 2026')
    expect(formatPeriodEndShort('2025-12-31')).toBe('Dec 2025')
    expect(formatPeriodEndShort('2025-13-01')).toBe('2025-13-01')
    expect(formatPeriodEndShort('FY2025')).toBe('FY2025')
    expect(formatPeriodEndShort(null)).toBe('')
  })

  it('maps every QueryIntentType to expected label, shortCode, and tone', () => {
    expect(getIntentDisplayMeta('multi_company_peer_comparison').shortCode).toBe('PEER COMP')
    expect(getIntentDisplayMeta('thematic_trend_tracking').shortCode).toBe('THEMATIC')
    expect(getIntentDisplayMeta('chunk_health_monitoring').shortCode).toBe('INDEX HEALTH')
    expect(getIntentDisplayMeta('safety_guardrail_block').tone).toBe('negative')
    expect(getIntentDisplayMeta('projection_compliance_block').shortCode).toBe('PROJECTION BLOCK')
    expect(getIntentDisplayMeta('clarification_needed').shortCode).toBe('CLARIFY')
    expect(getIntentDisplayMeta('out_of_corpus').shortCode).toBe('OUT OF CORPUS')
    expect(getIntentDisplayMeta('longitudinal_single_company').shortCode).toBe('LONGITUDINAL')
    expect(getIntentDisplayMeta(null).shortCode).toBe('LONGITUDINAL')
  })

  it('splits text into inline citations and XBRL tokens including dotted and hyphenated tickers', () => {
    const text =
      'Revenue rose ' +
      '[BRK.B FY2025 | XBRL us-gaap:Revenues | acc:0000950170-26-000123](https://sec.gov/brk) ' +
      'with commentary ' +
      '[BRK-B FY2025 | Item 7. MD&A | BRK-B_2025_item_7_0001](https://sec.gov/brk10k) trailing.'

    const segs = splitTextByInlineCitations(text)
    expect(segs).toHaveLength(5)
    expect(segs[0]).toEqual({ type: 'text', value: 'Revenue rose ' })
    expect(segs[1]).toEqual({
      type: 'xbrl',
      token: {
        rawMatch: '[BRK.B FY2025 | XBRL us-gaap:Revenues | acc:0000950170-26-000123](https://sec.gov/brk)',
        ticker: 'BRK.B',
        fiscalYear: 2025,
        xbrlConcept: 'us-gaap:Revenues',
        accessionNumber: '0000950170-26-000123',
        url: 'https://sec.gov/brk',
      },
    })
    expect(segs[3]).toEqual({
      type: 'citation',
      token: {
        rawMatch: '[BRK-B FY2025 | Item 7. MD&A | BRK-B_2025_item_7_0001](https://sec.gov/brk10k)',
        ticker: 'BRK-B',
        fiscalYear: 2025,
        sectionLabel: 'Item 7. MD&A',
        chunkId: 'BRK-B_2025_item_7_0001',
        url: 'https://sec.gov/brk10k',
      },
    })
    expect(segs[4]).toEqual({ type: 'text', value: ' trailing.' })
  })
})

describe('xbrlFactSelection builders', () => {
  const mv: MetricVariance = {
    metric_name: 'revenue',
    start_year: 2024,
    end_year: 2026,
    start_value: 60_922_000_000,
    end_value: 215_938_000_000,
    absolute_delta: 155_016_000_000,
    percentage_delta: 254.45,
    cagr_pct: 88.27,
    direction: 'increase',
  }

  const step: AnnualStepVariance = {
    fiscal_year: 2026,
    prev_fiscal_year: 2025,
    revenue: 215_938_000_000,
    revenue_yoy_delta: 85_441_000_000,
    revenue_yoy_pct: 65.47,
    operating_income: 130_387_000_000,
    operating_income_yoy_delta: 48_934_000_000,
    operating_income_yoy_pct: 60.08,
    net_income: 120_067_000_000,
    net_income_yoy_delta: 47_187_000_000,
    net_income_yoy_pct: 64.75,
    sec_filing_url: 'https://www.sec.gov/nvda-2026.htm',
  }

  const rep: CompanyVarianceReport = {
    ticker: 'NVDA',
    company_name: 'NVIDIA CORP',
    cik: '0001045810',
    gics_sector: 'Information Technology',
    gics_sub_industry: 'Semiconductors',
    start_year: 2024,
    end_year: 2026,
    currency: 'USD',
    revenue_variance: mv,
    operating_income_variance: { ...mv, metric_name: 'operating_income' },
    net_income_variance: { ...mv, metric_name: 'net_income' },
    operating_margin_bps_delta: 626,
    net_margin_bps_delta: 675,
    annual_series: [step],
    source_urls: {
      '2024': 'https://www.sec.gov/nvda-2024.htm',
      '2026': 'https://www.sec.gov/nvda-2026.htm',
    },
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
  }

  it('builds XBRL fact selections from variance, annual step, flag, and inline token with fallbacks', () => {
    const fromVar = buildXbrlFactFromVariance(rep, mv, 'Revenue')
    expect(fromVar.title).toBe('NVDA Revenue (FY2024→FY2026)')
    expect(fromVar.valueFormatted).toBe('$60.92B → $215.94B')
    expect(fromVar.deltaFormatted).toBe('+254.45% (+$155.02B)')
    expect(fromVar.accessionNumber).toBe('0001045810-26-000021')
    expect(fromVar.priorAccessionNumber).toBe('0001045810-24-000029')
    expect(fromVar.xbrlConcept).toBe('us-gaap:Revenues')

    const fromStep = buildXbrlFactFromAnnualStep(rep, step)
    expect(fromStep.title).toBe('NVDA FY2026 Annual Filing Facts')
    expect(fromStep.deltaFormatted).toBe('+65.47% YoY')
    expect(fromStep.periodOfReport).toBe('2026-01-25')
    expect(fromStep.accessionNumber).toBe('0001045810-26-000021')

    const flag: MaterialVarianceFlag = {
      flag_id: 'INTC_2024_net_income_sign_reversal',
      ticker: 'INTC',
      company_name: 'INTEL CORP',
      gics_sector: 'Information Technology',
      fiscal_year: 2024,
      prior_fiscal_year: 2023,
      metric: 'net_income',
      rule: 'sign_reversal',
      severity: 'high',
      value: -18_756_000_000,
      unit: 'usd',
      threshold: 0,
      current_value: -18_756_000_000,
      prior_value: 1_689_000_000,
      benchmark_value: null,
      peer_count: null,
      description: 'Swung from profit to loss',
      explanation_query: 'Intel FY2024 net loss',
      explanation: null,
      source: {
        accession_number: '0000050863-25-000009',
        prior_accession_number: '0000050863-24-000010',
        period_of_report: '2024-12-28',
        prior_period_of_report: '2023-12-30',
        xbrl_concept: 'us-gaap:NetIncomeLoss',
        sec_filing_url: 'https://www.sec.gov/intc-2024.htm',
        prior_sec_filing_url: 'https://www.sec.gov/intc-2023.htm',
      },
    }
    const fromFlag = buildXbrlFactFromFlag(flag, 'Net Income', 'Sign Reversal', 'Swing to Loss')
    expect(fromFlag.title).toBe('INTC Net Income (Sign Reversal)')
    expect(fromFlag.accessionNumber).toBe('0000050863-25-000009')

    const fromToken = buildXbrlFactFromInlineToken(
      {
        rawMatch: '',
        ticker: 'NVDA',
        fiscalYear: 2026,
        xbrlConcept: 'us-gaap:Revenues',
        accessionNumber: '0001045810-26-000021',
        url: '',
      },
      rep
    )
    expect(fromToken.companyName).toBe('NVIDIA CORP')
    expect(fromToken.periodOfReport).toBe('2026-01-25')
    expect(fromToken.secFilingUrl).toBe('https://www.sec.gov/nvda-2026.htm')
  })
})
