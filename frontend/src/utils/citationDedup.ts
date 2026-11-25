import type { GroundedCitation } from '../types/edgar'

/**
 * Group backend-deduplicated grounded passages for display.
 *
 * The backend (`deduplicate_citations_by_text`) collapses duplicate passages
 * and attaches merged filings under `duplicate_sources`. This helper expands
 * `duplicate_sources` into group members and builds `aliasOf` so inline
 * citations targeting merged chunk IDs still highlight the primary card.
 */

export interface CitationGroup {
  primary: GroundedCitation
  /** Other filings whose passage text duplicates the primary's. */
  duplicates: GroundedCitation[]
}

export interface DedupedCitations {
  groups: CitationGroup[]
  /** Every member chunk ID → its group's primary chunk ID. */
  aliasOf: Record<string, string>
}

export function dedupeCitations(citations: GroundedCitation[]): DedupedCitations {
  const groups: CitationGroup[] = []
  const aliasOf: Record<string, string> = {}

  for (const cit of citations ?? []) {
    if (aliasOf[cit.chunk_id]) {
      continue
    }
    const backendDups: GroundedCitation[] = (cit.duplicate_sources ?? []).map((ds) => ({
      ...cit,
      chunk_id: ds.chunk_id,
      ticker: ds.ticker,
      fiscal_year: ds.fiscal_year,
      section_id: ds.section_id,
      section_title: ds.section_title,
      sec_filing_url: ds.sec_filing_url,
      duplicate_sources: undefined,
    }))
    groups.push({ primary: cit, duplicates: backendDups })
    aliasOf[cit.chunk_id] = cit.chunk_id
    for (const ds of backendDups) {
      aliasOf[ds.chunk_id] = cit.chunk_id
    }
  }

  return { groups, aliasOf }
}

/** Number of distinct passages after de-duplication. */
export function countDistinctCitations(citations: GroundedCitation[]): number {
  return dedupeCitations(citations).groups.length
}
