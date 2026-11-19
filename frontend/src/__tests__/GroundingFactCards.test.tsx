import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { DisclosureDeltaCard } from '../components/DisclosureDeltaCard'
import { MaterialFlagsCard } from '../components/MaterialFlagsCard'
import type {
  DisclosureDeltaResult,
  GroundedCitation,
  MaterialVarianceFlag,
} from '../types/edgar'

function citation(chunkId: string, fy: number, section: 'item_1a' | 'item_7'): GroundedCitation {
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
    sec_filing_url: `https://www.sec.gov/nvda/${fy}.htm`,
    unstructured_gcs_uri: '',
    score: 0.9,
  }
}

describe('Grounding fact cards', () => {
  afterEach(() => cleanup())

  it('renders material variance flags and opens the MD&A explanation citation', () => {
    const flag: MaterialVarianceFlag = {
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
        sec_filing_url: 'https://www.sec.gov/nvda/2025.htm',
        period_of_report: '2025-01-26',
        xbrl_concept: 'us-gaap:Revenues',
        prior_accession_number: null,
        prior_sec_filing_url: null,
        prior_period_of_report: null,
      },
      explanation: {
        chunk_id: 'NVDA_2025_item_7_0004',
        section_id: 'item_7',
        section_title: 'Item 7. MD&A',
        excerpt: 'Data Center revenue grew on Hopper demand.',
        sec_filing_url: 'https://www.sec.gov/nvda/2025.htm',
      },
    }
    const onSelect = vi.fn()
    const onInspectXbrl = vi.fn()
    render(
      <MaterialFlagsCard
        turnId="t1"
        flags={[flag]}
        totalFlags={4}
        citations={[citation('NVDA_2025_item_7_0004', 2025, 'item_7')]}
        onSelectCitation={onSelect}
        onInspectXbrlFact={onInspectXbrl}
      />
    )
    expect(screen.getByText('Material Variance Flags')).toBeTruthy()
    expect(screen.getByText('+114.20%')).toBeTruthy()
    expect(screen.getByText(/1 OF 4/)).toBeTruthy()
    expect(screen.getByText('2025-01-26')).toBeTruthy()
    expect(screen.getByText('us-gaap:Revenues')).toBeTruthy()
    fireEvent.click(screen.getByText('[XBRL]'))
    expect(onInspectXbrl).toHaveBeenCalledWith(
      expect.objectContaining({
        accessionNumber: '0001045810-25-000023',
        xbrlConcept: 'us-gaap:Revenues',
      })
    )
    fireEvent.click(screen.getByText('[MD&A WHY]'))
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ chunkId: 'NVDA_2025_item_7_0004', fiscalYear: 2025 })
    )
  })

  it('renders disclosure delta counts and cites both years of a modified passage', () => {
    const result: DisclosureDeltaResult = {
      status: 'success',
      theme: 'AI risk',
      summary: {
        added: 1,
        removed: 0,
        modified: 1,
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
          missing_fiscal_years: [2026],
          theme_mentions: { '2024': 47, '2025': 64 },
          year_pairs: [
            {
              prior_fiscal_year: 2024,
              current_fiscal_year: 2025,
              counts: { added: 1, modified: 1, removed: 0, unchanged: 0 },
              deltas: [
                {
                  delta_id: 'd1',
                  ticker: 'NVDA',
                  company_name: 'NVIDIA CORP',
                  section_id: 'item_1a',
                  section_title: 'Item 1A. Risk Factors',
                  change_type: 'modified',
                  prior_fiscal_year: 2024,
                  current_fiscal_year: 2025,
                  prior_chunk_id: 'NVDA_2024_item_1a_0046',
                  current_chunk_id: 'NVDA_2025_item_1a_0048',
                  prior_excerpt: 'Old export wording.',
                  current_excerpt: 'New AI Diffusion IFR wording.',
                  similarity: 0.7,
                  added_phrases: ['AI Diffusion IFR'],
                  removed_phrases: [],
                  theme_relevance: 0.8,
                },
              ],
            },
          ],
        },
      ],
    }
    const onSelect = vi.fn()
    render(
      <DisclosureDeltaCard
        turnId="t2"
        result={result}
        citations={[
          citation('NVDA_2024_item_1a_0046', 2024, 'item_1a'),
          citation('NVDA_2025_item_1a_0048', 2025, 'item_1a'),
        ]}
        onSelectCitation={onSelect}
      />
    )
    expect(screen.getByText('Disclosure Changes YoY')).toBeTruthy()
    expect(screen.getByText('FY24:47 → FY25:64')).toBeTruthy()
    expect(screen.getByText(/no FY26/)).toBeTruthy()
    fireEvent.click(screen.getByText(/FY24→FY25/))
    expect(screen.getByText('New AI Diffusion IFR wording.')).toBeTruthy()
    fireEvent.click(screen.getByText('[FY2024]'))
    fireEvent.click(screen.getByText('[FY2025]'))
    expect(onSelect).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ chunkId: 'NVDA_2024_item_1a_0046' })
    )
    expect(onSelect).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ chunkId: 'NVDA_2025_item_1a_0048' })
    )
  })
})
