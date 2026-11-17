import type { GroundedCitation } from '../types/edgar'
import type { ParsedInlineCitationToken } from './formatters'

/** Build the inline-citation token the source pane understands from a citation record. */
export function citationToken(cit: GroundedCitation): ParsedInlineCitationToken {
  const sectionLabel =
    cit.section_id === 'item_1a' ? 'Item 1A Risk Factors' : 'Item 7 MD&A'
  return {
    rawMatch: `[${cit.ticker} FY${cit.fiscal_year} | ${sectionLabel} | ${cit.chunk_id}](${cit.sec_filing_url})`,
    ticker: cit.ticker,
    fiscalYear: cit.fiscal_year,
    sectionLabel,
    chunkId: cit.chunk_id,
    url: cit.sec_filing_url,
  }
}
