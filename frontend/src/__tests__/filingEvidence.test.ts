import { describe, expect, it } from 'vitest'
import type { GroundedCitation } from '../types/edgar'
import { countDistinctCitations, dedupeCitations } from '../utils/citationDedup'
import {
  compactTableCells,
  filingHighlightAnchor,
  filingPassagePreview,
  isNumericCell,
  parseFilingMarkdown,
  stripChunkContextHeader,
} from '../utils/filingMarkdown'
import {
  describeSectionLabel,
  humanizeXbrlConcept,
  shortCitationTopic,
} from '../utils/xbrlLabels'

// Verbatim shape of a corpus chunk produced by vector_db/chunker.py.
const AAPL_SEGMENT_CHUNK = [
  '[AAPL (Apple Inc.) | FY2024 Form 10-K | Item 7. MD&A — Segment Operating Performance]',
  '',
  '**Segment Operating Performance**',
  '',
  'The following table shows net sales by reportable segment for 2024, 2023 and 2022 \\(dollars in millions\\):',
  '',
  '|  | 2024 |  | Change |  | 2023 |  | Change |  | 2022 |  |',
  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
  '| Americas | $ | 167,045 | 3 | % | $ | 162,560 | (4) | % | $ | 169,658 |',
  '| Europe | 101,328 |  | 7 | % | 94,294 |  | (1) | % | 95,118 |  |',
  '',
  '*Americas*',
  '',
  'Americas net sales increased during 2024 compared to 2023 due primarily to higher net sales of Services\\.',
  '',
  'Apple Inc\\. | 2024 Form 10\\-K | 22',
].join('\n')

const AAPL_BULLET_CHUNK = [
  '[AAPL (Apple Inc.) | FY2024 Form 10-K | Item 7. MD&A — Product Announcements]',
  '',
  'Significant announcements during fiscal year 2024 included the following:',
  '',
  '• MacBook Pro 14\\-in\\.;',
  '',
  '• iMac\\.',
  '',
  'Second Quarter 2024:',
].join('\n')

function cit(
  chunkId: string,
  excerpt: string,
  overrides: Partial<GroundedCitation> = {}
): GroundedCitation {
  return {
    chunk_id: chunkId,
    profile_id: 'NVDA_2025',
    ticker: 'NVDA',
    company_name: 'NVIDIA CORP',
    gics_sector: 'Information Technology',
    fiscal_year: 2025,
    section_id: 'item_1a',
    section_title: 'Item 1A. Risk Factors',
    excerpt,
    sec_filing_url: 'https://www.sec.gov/nvda.htm',
    unstructured_gcs_uri: '',
    score: 0.5,
    ...overrides,
  }
}

const RISK_TEXT =
  'Our operations could be adversely affected by export controls imposed by the United States government on sales of our data center products to customers in China and other regions, which may reduce demand and require us to redesign products at significant cost.'

describe('humanizeXbrlConcept', () => {
  it('maps verbose us-gaap concepts to analyst labels', () => {
    expect(
      humanizeXbrlConcept(
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
      )
    ).toBe('Pre-tax Income')
    expect(humanizeXbrlConcept('us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax')).toBe('Revenue')
    expect(humanizeXbrlConcept('us-gaap:NetIncomeLoss')).toBe('Net Income')
    expect(humanizeXbrlConcept('us-gaap:OperatingIncomeLoss')).toBe('Operating Income')
  })

  it('unwraps provenance strings and issuer-extension namespaces', () => {
    expect(
      humanizeXbrlConcept('edgartools:Financials.get_operating_income (us-gaap:OperatingIncomeLoss)')
    ).toBe('Operating Income')
    expect(humanizeXbrlConcept('edgartools:Financials.get_revenue')).toBe('Revenue')
    expect(
      humanizeXbrlConcept('sec-companyfacts:us-gaap:ResearchAndDevelopmentExpense')
    ).toBe('R&D Expense')
    expect(
      humanizeXbrlConcept('acgl:IncomeLossFromContinuingOperationsBeforeEarningsFromAffiliatesIncomeTaxesNoncontrollingInterest')
    ).toBe('Pre-tax Income')
  })

  it('falls back to a short word label for unknown concepts', () => {
    expect(humanizeXbrlConcept('us-gaap:AccountsPayableCurrent')).toBe('Accounts Payable Current')
    expect(humanizeXbrlConcept('xyz:SomeVeryLongCustomConceptNameHere')).toBe('Some Very Long Custom…')
    expect(humanizeXbrlConcept('')).toBe('Reported Figure')
  })
})

describe('section labels', () => {
  it('splits breadcrumbs into section and topic', () => {
    expect(describeSectionLabel('Item 7. MD&A — Gross Margin')).toEqual({
      section: 'MD&A',
      topic: 'Gross Margin',
    })
    expect(describeSectionLabel('Item 1A Risk Factors')).toEqual({
      section: 'Risk Factors',
      topic: null,
    })
  })

  it('uses short topics in chips and the section name for long risk headings', () => {
    expect(shortCitationTopic('Item 7. MD&A — Gross Margin')).toBe('Gross Margin')
    expect(
      shortCitationTopic(
        'Item 1A. Risk Factors — A temporary or permanent blockchain fork could adversely affect our business.'
      )
    ).toBe('Risk Factors')
  })
})

describe('parseFilingMarkdown', () => {
  it('strips the context header, escapes and page footers', () => {
    expect(stripChunkContextHeader(AAPL_SEGMENT_CHUNK).startsWith('**Segment')).toBe(true)
    const blocks = parseFilingMarkdown(AAPL_SEGMENT_CHUNK)
    const text = JSON.stringify(blocks)
    expect(text).not.toContain('Form 10-K | 22')
    expect(text).not.toContain('\\\\')
    expect(blocks[0]).toEqual({ kind: 'heading', level: 2, text: 'Segment Operating Performance' })
    expect(blocks[1]).toEqual({
      kind: 'paragraph',
      text: 'The following table shows net sales by reportable segment for 2024, 2023 and 2022 (dollars in millions):',
    })
    expect(blocks).toContainEqual({ kind: 'heading', level: 3, text: 'Americas' })
  })

  it('repairs SEC tables with split $ and % cells', () => {
    const table = parseFilingMarkdown(AAPL_SEGMENT_CHUNK).find((b) => b.kind === 'table')
    expect(table).toEqual({
      kind: 'table',
      header: ['', '2024', 'Change', '2023', 'Change', '2022'],
      rows: [
        { cells: ['Americas', '$167,045', '3%', '$162,560', '(4)%', '$169,658'], isSection: false },
        { cells: ['Europe', '101,328', '7%', '94,294', '(1)%', '95,118'], isSection: false },
      ],
    })
    expect(compactTableCells(['Loss', '$', '(1,234', ')'])).toEqual(['Loss', '$(1,234)'])
    expect(isNumericCell('$(1,234)')).toBe(true)
    expect(isNumericCell('(4)%')).toBe(true)
    expect(isNumericCell('Recovers fuel costs')).toBe(false)
  })

  it('groups blank-line separated bullets into a single list', () => {
    const blocks = parseFilingMarkdown(AAPL_BULLET_CHUNK)
    expect(blocks).toEqual([
      { kind: 'paragraph', text: 'Significant announcements during fiscal year 2024 included the following:' },
      { kind: 'list', items: ['MacBook Pro 14-in.;', 'iMac.'] },
      { kind: 'paragraph', text: 'Second Quarter 2024:' },
    ])
  })

  it('builds a plain preview that skips a heading repeating the topic', () => {
    expect(filingPassagePreview(AAPL_SEGMENT_CHUNK, 'Segment Operating Performance')).toBe(
      'The following table shows net sales by reportable segment for 2024, 2023 and 2022 (dollars in millions):'
    )
    expect(filingPassagePreview(AAPL_SEGMENT_CHUNK, null)).toBe('Segment Operating Performance')
  })

  it('picks a prose paragraph (not the header) to locate the passage in the full 10-K', () => {
    const anchor = filingHighlightAnchor(AAPL_SEGMENT_CHUNK)
    expect(anchor.startsWith('The following table shows net sales')).toBe(true)
  })
})

describe('dedupeCitations', () => {
  it('collapses repeated chunk IDs and identical passages under different IDs', () => {
    const a = cit('NVDA_2025_item_1a_0004', `[NVDA (NVIDIA) | FY2025 Form 10-K | Item 1A]\n\n${RISK_TEXT}`)
    const sameId = { ...a }
    const nextYear = cit('NVDA_2026_item_1a_0005', `[NVDA (NVIDIA) | FY2026 Form 10-K | Item 1A]\n\n${RISK_TEXT}`, {
      fiscal_year: 2026,
    })
    const different = cit('NVDA_2026_item_7_0002', 'Revenue for fiscal year 2026 was $215.9 billion, up 65% from a year ago.', {
      section_id: 'item_7',
    })

    const result = dedupeCitations([a, sameId, nextYear, different])
    expect(result.groups).toHaveLength(2)
    expect(result.groups[0].primary.chunk_id).toBe('NVDA_2025_item_1a_0004')
    expect(result.groups[0].duplicates.map((d) => d.chunk_id)).toEqual(['NVDA_2026_item_1a_0005'])
    expect(result.aliasOf['NVDA_2026_item_1a_0005']).toBe('NVDA_2025_item_1a_0004')
    expect(countDistinctCitations([a, sameId, nextYear, different])).toBe(2)
  })

  it('merges near-identical boilerplate within a section and keeps the fuller excerpt', () => {
    const base = `${RISK_TEXT} ${RISK_TEXT.replace('Our operations', 'Our results')} These risks are described further in Item 7 and in the notes to our consolidated financial statements included elsewhere in this annual report.`
    const tweaked = base.replace('significant cost', 'substantial cost')
    const clipped = cit('NVDA_2024_item_1a_0003', base.slice(0, 180), { fiscal_year: 2024 })
    const full = cit('NVDA_2025_item_1a_0004', base)
    const nearDup = cit('NVDA_2026_item_1a_0004', tweaked, { fiscal_year: 2026 })

    const { groups, aliasOf } = dedupeCitations([clipped, full, nearDup])
    expect(groups).toHaveLength(1)
    expect(groups[0].primary.chunk_id).toBe('NVDA_2025_item_1a_0004')
    expect(aliasOf['NVDA_2024_item_1a_0003']).toBe('NVDA_2025_item_1a_0004')
    expect(aliasOf['NVDA_2026_item_1a_0004']).toBe('NVDA_2025_item_1a_0004')
  })

  it('does not merge similar text across different sections or distinct passages', () => {
    const risk = cit('A_item_1a', RISK_TEXT)
    const mdna = cit('A_item_7', RISK_TEXT.replace('significant cost', 'substantial cost'), {
      section_id: 'item_7',
    })
    const other = cit('B_item_1a', 'Our manufacturing depends on a limited number of foundry partners in Taiwan, and disruption there could materially harm supply.')
    expect(dedupeCitations([risk, mdna, other]).groups).toHaveLength(3)
  })
})
