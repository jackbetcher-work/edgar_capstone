import type { GroundedCitation } from '../types/edgar'
import { passageWordTokens } from './filingMarkdown'

/**
 * Collapse duplicate grounded passages for display.
 *
 * Retrieval merges results from several searches (per-ticker, boundary-year,
 * agent tool calls, disclosure-delta pairs), so the same evidence can surface
 * more than once:
 * - the same `chunk_id` returned by two code paths
 * - identical boilerplate repeated in consecutive 10-Ks (or in share-class
 *   twins such as GOOG/GOOGL) under different chunk IDs
 * - a clipped excerpt of a passage that is also present in full
 *
 * Each group keeps one primary card; the other members are kept as
 * `duplicates` so the analyst can still see every filing the passage
 * appears in, and `aliasOf` maps every member chunk ID to the primary so
 * inline citations still land on the right card.
 */

export interface CitationGroup {
  primary: GroundedCitation
  /** Other citations whose passage text duplicates the primary's. */
  duplicates: GroundedCitation[]
}

export interface DedupedCitations {
  groups: CitationGroup[]
  /** Every member chunk ID → its group's primary chunk ID. */
  aliasOf: Record<string, string>
}

/** Word 3-gram Jaccard similarity at/above which passages are treated as duplicates. */
export const NEAR_DUPLICATE_THRESHOLD = 0.9
/** A shorter passage fully contained in a longer one must have at least this many words. */
const MIN_CONTAINED_WORDS = 15

interface Fingerprint {
  joined: string
  shingles: Set<string>
  wordCount: number
}

function fingerprint(excerpt: string): Fingerprint {
  const tokens = passageWordTokens(excerpt)
  const shingles = new Set<string>()
  if (tokens.length < 3) {
    if (tokens.length > 0) shingles.add(tokens.join(' '))
  } else {
    for (let i = 0; i + 3 <= tokens.length; i += 1) {
      shingles.add(`${tokens[i]} ${tokens[i + 1]} ${tokens[i + 2]}`)
    }
  }
  return { joined: tokens.join(' '), shingles, wordCount: tokens.length }
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0
  let intersection = 0
  const [small, large] = a.size <= b.size ? [a, b] : [b, a]
  small.forEach((s) => {
    if (large.has(s)) intersection += 1
  })
  return intersection / (a.size + b.size - intersection)
}

function isDuplicatePassage(
  a: Fingerprint,
  b: Fingerprint,
  sameSection: boolean
): boolean {
  if (!a.joined || !b.joined) return false
  if (a.joined === b.joined) return true
  const [shorter, longer] = a.wordCount <= b.wordCount ? [a, b] : [b, a]
  if (shorter.wordCount >= MIN_CONTAINED_WORDS && longer.joined.includes(shorter.joined)) {
    return true
  }
  return sameSection && jaccard(a.shingles, b.shingles) >= NEAR_DUPLICATE_THRESHOLD
}

export function dedupeCitations(citations: GroundedCitation[]): DedupedCitations {
  const groups: Array<CitationGroup & { fp: Fingerprint }> = []
  const aliasOf: Record<string, string> = {}

  for (const cit of citations ?? []) {
    const existingPrimary = aliasOf[cit.chunk_id]
    if (existingPrimary) {
      // Same chunk_id returned twice: nothing new to show.
      continue
    }
    const fp = fingerprint(cit.excerpt ?? '')
    const match = groups.find((g) =>
      isDuplicatePassage(g.fp, fp, g.primary.section_id === cit.section_id)
    )
    if (!match) {
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
      groups.push({ primary: cit, duplicates: backendDups, fp })
      aliasOf[cit.chunk_id] = cit.chunk_id
      for (const ds of backendDups) {
        aliasOf[ds.chunk_id] = cit.chunk_id
      }
      continue
    }
    // Prefer the fuller excerpt as the visible text, keeping the group's position.
    if (fp.wordCount > match.fp.wordCount) {
      const previousPrimaryId = match.primary.chunk_id
      match.duplicates.push(match.primary)
      match.primary = cit
      match.fp = fp
      Object.keys(aliasOf).forEach((id) => {
        if (aliasOf[id] === previousPrimaryId) aliasOf[id] = cit.chunk_id
      })
    } else {
      match.duplicates.push(cit)
    }
    aliasOf[cit.chunk_id] = match.primary.chunk_id
  }

  return {
    groups: groups.map(({ primary, duplicates }) => ({ primary, duplicates })),
    aliasOf,
  }
}

/** Number of distinct passages after de-duplication. */
export function countDistinctCitations(citations: GroundedCitation[]): number {
  return dedupeCitations(citations).groups.length
}
