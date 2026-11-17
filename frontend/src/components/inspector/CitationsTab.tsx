import { useEffect, useMemo, useRef } from 'react'
import { ExternalLink, FileText } from 'lucide-react'
import type { GroundedCitation } from '../../types/edgar'
import { clean10kExcerpt } from '../../utils/formatters'

export interface FilingTarget {
  ticker: string
  fiscalYear: number
}

export type CitationSectionFilter = 'ALL' | 'item_7' | 'item_1a'

interface CitationsTabProps {
  citations: GroundedCitation[]
  activeChunkId: string | null
  sectionFilter: CitationSectionFilter
  onChangeSectionFilter: (filter: CitationSectionFilter) => void
  onSelectChunkId: (chunkId: string | null) => void
  filingTargets: FilingTarget[]
  onOpenFilingSection: (
    ticker: string,
    fiscalYear: number,
    sectionId: 'item_7' | 'item_1a',
    highlightExcerpt?: string
  ) => void
}

export function CitationsTab({
  citations,
  activeChunkId,
  sectionFilter,
  onChangeSectionFilter,
  onSelectChunkId,
  filingTargets,
  onOpenFilingSection,
}: CitationsTabProps) {
  const citationCardRefs = useRef<Record<string, HTMLElement | null>>({})

  useEffect(() => {
    if (activeChunkId) {
      const targetEl = citationCardRefs.current[activeChunkId]
      if (targetEl && typeof targetEl.scrollIntoView === 'function') {
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      }
    }
  }, [activeChunkId])

  const filteredCitations = useMemo(() => {
    if (sectionFilter === 'ALL') return citations
    return citations.filter((c) => c.section_id === sectionFilter)
  }, [citations, sectionFilter])

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="px-3 py-1.5 border-b border-hairline flex items-center justify-between gap-2 surface-subtle tabular-nums-mono text-[10px] uppercase shrink-0">
        <span className="text-ink-muted">SOURCE PASSAGES</span>
        <div className="flex items-center gap-1">
          {(['ALL', 'item_7', 'item_1a'] as const).map((secKey) => (
            <button
              key={secKey}
              id={`btn-filter-cit-${secKey}`}
              type="button"
              onClick={() => onChangeSectionFilter(secKey)}
              className={`px-1.5 py-0.2 cursor-pointer transition-colors ${
                sectionFilter === secKey
                  ? 'badge-amber font-semibold'
                  : 'text-ink-muted hover:text-ink-primary'
              }`}
            >
              {secKey === 'ALL'
                ? 'ALL'
                : secKey === 'item_7'
                  ? 'ITEM 7'
                  : 'ITEM 1A'}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto divide-y divide-[var(--border-hairline)]">
        {filteredCitations.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-6 py-10 tabular-nums-mono">
            <FileText
              className="w-6 h-6 text-ink-muted mb-2 stroke-[1.5]"
              aria-hidden="true"
            />
            <p className="text-xs font-semibold uppercase tracking-wider text-ink-primary">
              NO 10-K PASSAGES LOADED
            </p>
            <p className="text-[11px] text-ink-muted mt-1 leading-relaxed max-w-xs font-sans">
              Execute a query or click an inline citation tag to inspect
              grounded Form 10-K excerpts here.
            </p>
            {filingTargets[0] && (
              <button
                id="btn-browse-sample-10k"
                type="button"
                onClick={() =>
                  onOpenFilingSection(
                    filingTargets[0].ticker,
                    filingTargets[0].fiscalYear,
                    'item_7'
                  )
                }
                className="mt-3 px-2.5 py-1 text-[11px] uppercase tracking-wider btn-terminal-key cursor-pointer"
              >
                OPEN {filingTargets[0].ticker} FY{filingTargets[0].fiscalYear}{' '}
                10-K
              </button>
            )}
          </div>
        ) : (
          filteredCitations.map((cit, index) => {
            const isSelected = activeChunkId === cit.chunk_id
            const isRisk = cit.section_id === 'item_1a'
            return (
              <div
                key={cit.chunk_id}
                id={`source-card-${cit.chunk_id}`}
                ref={(el) => {
                  citationCardRefs.current[cit.chunk_id] = el
                }}
                onClick={() =>
                  onSelectChunkId(isSelected ? null : cit.chunk_id)
                }
                className={`p-3 transition-colors cursor-pointer ${
                  isSelected
                    ? 'surface-elevated border-l-2 border-l-[var(--bb-amber)]'
                    : 'hover:bg-[var(--bg-subtle)]'
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-1 tabular-nums-mono">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="text-[11px] text-ink-muted">
                      #{index + 1}
                    </span>
                    <span className="text-[11px] font-bold text-bb-yellow">
                      {cit.ticker}
                    </span>
                    <span className="text-[11px] text-ink-primary">
                      FY{cit.fiscal_year}
                    </span>
                    {cit.period_of_report && (
                      <span className="text-[10px] text-ink-muted">
                        ({cit.period_of_report})
                      </span>
                    )}
                    <span
                      className={`px-1.5 py-0.2 text-[10px] font-semibold uppercase ${
                        isRisk ? 'badge-amber' : 'badge-accent'
                      }`}
                    >
                      {isRisk ? 'ITEM 1A · RISK' : 'ITEM 7 · MD&A'}
                    </span>
                  </div>

                  <span
                    className="text-[10px] text-ink-muted"
                    title="Hybrid Retrieval Score"
                  >
                    REL {cit.score.toFixed(3)}
                  </span>
                </div>

                <div className="text-[11px] font-medium text-ink-secondary mb-1.5 truncate tabular-nums-mono">
                  {cit.section_title}
                </div>

                <blockquote
                  className={`filing-prose text-xs leading-relaxed p-2 border-l-2 ${
                    isSelected
                      ? 'passage-highlight border-l-[var(--bb-amber)]'
                      : 'surface-filing-sheet border-l-[var(--border-strong)]'
                  }`}
                >
                  &ldquo;{clean10kExcerpt(cit.excerpt)}&rdquo;
                </blockquote>

                <div className="mt-2 flex items-center justify-between gap-2 text-[11px] tabular-nums-mono">
                  <span className="text-[10px] text-ink-muted truncate">
                    {cit.chunk_id}
                  </span>

                  <div className="flex items-center gap-2.5 shrink-0">
                    <button
                      id={`btn-read-in-10k-${cit.chunk_id}`}
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelectChunkId(cit.chunk_id)
                        onOpenFilingSection(
                          cit.ticker,
                          cit.fiscal_year,
                          cit.section_id,
                          cit.excerpt
                        )
                      }}
                      className="text-[11px] font-medium text-bb-cyan hover:underline cursor-pointer"
                    >
                      Open in 10-K Section
                    </button>

                    {cit.sec_filing_url && (
                      <a
                        id={`link-edgar-cit-${cit.chunk_id}`}
                        href={cit.sec_filing_url}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="inline-flex items-center gap-0.5 text-[10px] uppercase text-ink-muted hover:text-ink-primary"
                        title="Open official SEC EDGAR HTML archive"
                      >
                        <span>EDGAR</span>
                        <ExternalLink className="w-2.5 h-2.5" />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
