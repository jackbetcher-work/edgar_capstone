import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ChevronDown,
  ChevronRight,
  ChevronsDownUp,
  ChevronsUpDown,
  ExternalLink,
  FileText,
} from 'lucide-react'
import type { GroundedCitation } from '../../types/edgar'
import { dedupeCitations } from '../../utils/citationDedup'
import {
  filingHighlightAnchor,
  filingPassagePreview,
} from '../../utils/filingMarkdown'
import { formatPeriodEndShort } from '../../utils/formatters'
import { describeSectionLabel } from '../../utils/xbrlLabels'
import { FilingMarkdown } from '../FilingMarkdown'

export interface FilingTarget {
  ticker: string
  fiscalYear: number
}

export type CitationSectionFilter = 'ALL' | 'item_7' | 'item_1a'

interface CitationsTabProps {
  citations: GroundedCitation[]
  activeChunkId: string | null
  /** Incremented on every explicit "show me this chunk" request (inline citation click). */
  focusNonce?: number
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

const EMPTY_SET: ReadonlySet<string> = new Set()

export function CitationsTab({
  citations,
  activeChunkId,
  focusNonce = 0,
  sectionFilter,
  onChangeSectionFilter,
  onSelectChunkId,
  filingTargets,
  onOpenFilingSection,
}: CitationsTabProps) {
  const citationCardRefs = useRef<Record<string, HTMLElement | null>>({})
  const { groups, aliasOf } = useMemo(() => dedupeCitations(citations), [citations])
  const listKey = useMemo(
    () => groups.map((g) => g.primary.chunk_id).join('|'),
    [groups]
  )

  // Cards start collapsed for every new set of sources.
  const [expansion, setExpansion] = useState<{ listKey: string; ids: Set<string> }>({
    listKey: '',
    ids: new Set(),
  })
  const expandedIds: ReadonlySet<string> =
    expansion.listKey === listKey ? expansion.ids : EMPTY_SET
  const [dismissedFocusKey, setDismissedFocusKey] = useState<string | null>(null)
  const [viewingVariant, setViewingVariant] = useState<Record<string, string>>({})

  const activePrimaryId = activeChunkId ? aliasOf[activeChunkId] ?? activeChunkId : null
  const focusKey = activePrimaryId ? `${focusNonce}:${activePrimaryId}` : null
  const focusedId = focusKey && focusKey !== dismissedFocusKey ? activePrimaryId : null

  useEffect(() => {
    if (!activePrimaryId) return
    const targetEl = citationCardRefs.current[activePrimaryId]
    if (targetEl && typeof targetEl.scrollIntoView === 'function') {
      targetEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [activePrimaryId, focusNonce])

  const filteredGroups = useMemo(() => {
    if (sectionFilter === 'ALL') return groups
    return groups.filter((g) => g.primary.section_id === sectionFilter)
  }, [groups, sectionFilter])

  const duplicatesMerged = citations.length - groups.length
  const isExpanded = (id: string) => expandedIds.has(id) || focusedId === id
  const anyExpanded = filteredGroups.some((g) => isExpanded(g.primary.chunk_id))

  const setExpandedIds = (ids: Set<string>) => setExpansion({ listKey, ids })

  const toggleCard = (id: string) => {
    if (isExpanded(id)) {
      const next = new Set(expandedIds)
      next.delete(id)
      setExpandedIds(next)
      if (focusedId === id) setDismissedFocusKey(focusKey)
    } else {
      setExpandedIds(new Set(expandedIds).add(id))
      onSelectChunkId(id)
    }
  }

  const toggleAll = () => {
    if (anyExpanded) {
      setExpandedIds(new Set())
      if (focusKey) setDismissedFocusKey(focusKey)
    } else {
      setExpandedIds(new Set(filteredGroups.map((g) => g.primary.chunk_id)))
    }
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="px-3 py-1.5 border-b border-hairline flex items-center justify-between gap-2 surface-subtle tabular-nums-mono text-[10px] uppercase shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-ink-muted">SOURCE PASSAGES</span>
          {duplicatesMerged > 0 && (
            <span
              className="text-ink-muted normal-case truncate"
              title="Passages with the same text across filings are shown once"
            >
              · {duplicatesMerged} duplicate{duplicatesMerged === 1 ? '' : 's'} merged
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
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
              {secKey === 'ALL' ? 'ALL' : secKey === 'item_7' ? 'MD&A' : 'RISK'}
            </button>
          ))}
          {filteredGroups.length > 0 && (
            <button
              id="btn-toggle-all-citations"
              type="button"
              onClick={toggleAll}
              className="ml-1 p-0.5 text-ink-muted hover:text-ink-primary cursor-pointer"
              aria-label={anyExpanded ? 'Collapse all passages' : 'Expand all passages'}
              title={anyExpanded ? 'Collapse all' : 'Expand all'}
            >
              {anyExpanded ? (
                <ChevronsDownUp className="w-3.5 h-3.5" />
              ) : (
                <ChevronsUpDown className="w-3.5 h-3.5" />
              )}
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto divide-y divide-[var(--border-hairline)]">
        {filteredGroups.length === 0 ? (
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
          filteredGroups.map((group, index) => {
            const primaryId = group.primary.chunk_id
            const variants = [group.primary, ...group.duplicates]
            const shownId = viewingVariant[primaryId] ?? primaryId
            const cit = variants.find((v) => v.chunk_id === shownId) ?? group.primary
            const expanded = isExpanded(primaryId)
            const isSelected = activePrimaryId === primaryId
            const isRisk = cit.section_id === 'item_1a'
            const { topic } = describeSectionLabel(cit.section_title)
            const title = topic ?? (isRisk ? 'Risk Factors' : "Management's Discussion & Analysis")
            const bodyId = `source-body-${primaryId}`
            const periodLabel = formatPeriodEndShort(cit.period_of_report)

            return (
              <div
                key={primaryId}
                id={`source-card-${primaryId}`}
                ref={(el) => {
                  citationCardRefs.current[primaryId] = el
                }}
                className={`transition-colors border-l-2 ${
                  isSelected
                    ? 'surface-elevated border-l-[var(--bb-amber)]'
                    : 'border-l-transparent'
                }`}
              >
                <button
                  id={`btn-toggle-source-${primaryId}`}
                  type="button"
                  onClick={() => toggleCard(primaryId)}
                  aria-expanded={expanded}
                  aria-controls={bodyId}
                  className="w-full text-left px-3 py-2 flex items-start gap-2 cursor-pointer hover:bg-[var(--bg-subtle)]"
                  title={`${cit.ticker} FY${cit.fiscal_year} · ${cit.section_title}`}
                >
                  {expanded ? (
                    <ChevronDown className="w-3.5 h-3.5 mt-0.5 shrink-0 text-bb-amber" />
                  ) : (
                    <ChevronRight className="w-3.5 h-3.5 mt-0.5 shrink-0 text-ink-muted" />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 tabular-nums-mono">
                      <span className="text-[10px] text-ink-muted">{index + 1}</span>
                      <span className="text-[11px] font-bold text-bb-yellow">
                        {cit.ticker}
                      </span>
                      <span className="text-[11px] text-ink-primary">
                        FY{cit.fiscal_year}
                      </span>
                      <span
                        className={`px-1 py-0.2 text-[9.5px] font-semibold uppercase ${
                          isRisk ? 'badge-amber' : 'badge-accent'
                        }`}
                      >
                        {isRisk ? 'Risk' : 'MD&A'}
                      </span>
                      {group.duplicates.length > 0 && (
                        <span
                          className="text-[9.5px] text-ink-muted uppercase"
                          title={`Same passage also cited from: ${group.duplicates
                            .map((d) => `${d.ticker} FY${d.fiscal_year}`)
                            .join(', ')}`}
                        >
                          +{group.duplicates.length} filing
                          {group.duplicates.length === 1 ? '' : 's'}
                        </span>
                      )}
                    </div>
                    <div className="mt-0.5 font-sans text-[12.5px] font-medium text-ink-primary truncate">
                      {title}
                    </div>
                    {!expanded && (
                      <div className="mt-0.5 font-sans text-[11.5px] text-ink-muted line-clamp-1">
                        {filingPassagePreview(cit.excerpt, topic)}
                      </div>
                    )}
                  </div>
                </button>

                {expanded && (
                  <div id={bodyId} className="px-3 pb-3 pl-8">
                    {variants.length > 1 && (
                      <div className="mb-2 flex flex-wrap items-center gap-1 text-[10px] tabular-nums-mono">
                        <span className="text-ink-muted uppercase">Appears in:</span>
                        {variants.map((v) => (
                          <button
                            key={v.chunk_id}
                            type="button"
                            onClick={() =>
                              setViewingVariant((prev) => ({
                                ...prev,
                                [primaryId]: v.chunk_id,
                              }))
                            }
                            className={`px-1.5 py-0.2 cursor-pointer ${
                              v.chunk_id === cit.chunk_id
                                ? 'badge-amber font-semibold'
                                : 'btn-terminal-key'
                            }`}
                            title={v.section_title}
                          >
                            {v.ticker} FY{v.fiscal_year}
                          </button>
                        ))}
                      </div>
                    )}

                    <div className="p-2.5 surface-filing-sheet border border-hairline">
                      <FilingMarkdown markdown={cit.excerpt} omitLeadingHeading={topic} />
                    </div>

                    <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] tabular-nums-mono">
                      <span className="text-[10px] text-ink-muted" title={cit.chunk_id}>
                        {cit.company_name}
                        {periodLabel ? ` · Period ended ${periodLabel}` : ''}
                      </span>
                      <div className="flex items-center gap-2.5 shrink-0">
                        <button
                          id={`btn-read-in-10k-${cit.chunk_id}`}
                          type="button"
                          onClick={() => {
                            onSelectChunkId(primaryId)
                            onOpenFilingSection(
                              cit.ticker,
                              cit.fiscal_year,
                              cit.section_id,
                              filingHighlightAnchor(cit.excerpt)
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
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
