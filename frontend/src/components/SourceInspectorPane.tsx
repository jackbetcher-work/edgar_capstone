import { useEffect, useMemo, useRef, useState } from 'react'
import {
  BookOpen,
  ExternalLink,
  FileCheck2,
  FileText,
  Layers,
  Loader2,
  Search,
  X,
} from 'lucide-react'
import type {
  CachedSectionResponse,
  CompanySummary,
  GroundedCitation,
} from '../types/edgar'
import { clean10kDocumentBlock, clean10kExcerpt } from '../utils/formatters'

interface SourceInspectorPaneProps {
  citations: GroundedCitation[]
  companies: CompanySummary[]
  activeChunkId: string | null
  onSelectChunkId: (chunkId: string | null) => void
  activeTab: 'citations' | 'document'
  onChangeTab: (tab: 'citations' | 'document') => void
  cachedSection: CachedSectionResponse | null
  isLoadingSection: boolean
  onRequestFilingSection: (
    ticker: string,
    fiscalYear: number,
    sectionId: 'item_7' | 'item_1a',
    highlightExcerpt?: string
  ) => void
  highlightedExcerpt: string | null
  onCloseMobilePane: () => void
}

function highlightTextSnippet(
  text: string,
  targetSnippet: string | null,
  searchKeyword: string
): Array<{ text: string; active: boolean; matchedSearch: boolean }> {
  const cleaned = clean10kDocumentBlock(text)
  const cleanTarget = targetSnippet ? clean10kExcerpt(targetSnippet).slice(0, 95) : ''
  const cleanKeyword = searchKeyword.trim()

  if (cleanTarget && cleaned.toLowerCase().includes(cleanTarget.toLowerCase())) {
    const idx = cleaned.toLowerCase().indexOf(cleanTarget.toLowerCase())
    const matchLen = Math.min(
      cleaned.length - idx,
      Math.max(cleanTarget.length, targetSnippet ? clean10kExcerpt(targetSnippet).length : 0)
    )
    return [
      { text: cleaned.slice(0, idx), active: false, matchedSearch: false },
      { text: cleaned.slice(idx, idx + matchLen), active: true, matchedSearch: false },
      { text: cleaned.slice(idx + matchLen), active: false, matchedSearch: false },
    ]
  }

  if (cleanKeyword.length >= 2) {
    const escaped = cleanKeyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const regex = new RegExp(`(${escaped})`, 'gi')
    const parts = cleaned.split(regex)
    return parts
      .filter((p) => p.length > 0)
      .map((part) => ({
        text: part,
        active: false,
        matchedSearch: part.toLowerCase() === cleanKeyword.toLowerCase(),
      }))
  }

  return [{ text: cleaned, active: false, matchedSearch: false }]
}

export function SourceInspectorPane({
  citations,
  companies,
  activeChunkId,
  onSelectChunkId,
  activeTab,
  onChangeTab,
  cachedSection,
  isLoadingSection,
  onRequestFilingSection,
  highlightedExcerpt,
  onCloseMobilePane,
}: SourceInspectorPaneProps) {
  const [citationSectionFilter, setCitationSectionFilter] = useState<
    'ALL' | 'item_7' | 'item_1a'
  >('ALL')
  const [docKeyword, setDocKeyword] = useState('')
  const citationCardRefs = useRef<Record<string, HTMLElement | null>>({})
  const highlightedDocMarkRef = useRef<HTMLElement | null>(null)

  // Scroll to active citation card when selected inline in chat
  useEffect(() => {
    if (activeTab === 'citations' && activeChunkId) {
      const targetEl = citationCardRefs.current[activeChunkId]
      if (targetEl && typeof targetEl.scrollIntoView === 'function') {
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      }
    }
  }, [activeChunkId, activeTab])

  // Scroll to highlighted passage inside Full 10-K Document reader
  useEffect(() => {
    if (activeTab === 'document' && highlightedExcerpt && !isLoadingSection) {
      const timer = setTimeout(() => {
        if (
          highlightedDocMarkRef.current &&
          typeof highlightedDocMarkRef.current.scrollIntoView === 'function'
        ) {
          highlightedDocMarkRef.current.scrollIntoView({
            behavior: 'smooth',
            block: 'center',
          })
        }
      }, 80)
      return () => clearTimeout(timer)
    }
  }, [activeTab, highlightedExcerpt, isLoadingSection, cachedSection?.cache_key])

  const filteredCitations = useMemo(() => {
    if (citationSectionFilter === 'ALL') return citations
    return citations.filter((c) => c.section_id === citationSectionFilter)
  }, [citations, citationSectionFilter])

  // Distinct company-year profiles from current citations (plus defaults so reader is always usable)
  const filingTargets = useMemo(() => {
    const map = new Map<string, { ticker: string; fiscalYear: number }>()
    citations.forEach((c) => {
      const key = `${c.ticker}_${c.fiscal_year}`
      if (!map.has(key)) {
        map.set(key, { ticker: c.ticker, fiscalYear: c.fiscal_year })
      }
    })
    if (map.size === 0) {
      companies.slice(0, 6).forEach((comp) => {
        map.set(`${comp.ticker}_${comp.latest_fiscal_year}`, {
          ticker: comp.ticker,
          fiscalYear: comp.latest_fiscal_year,
        })
      })
    }
    return Array.from(map.values())
  }, [citations, companies])

  const documentParagraphs = useMemo(() => {
    if (!cachedSection?.content) return []
    return cachedSection.content
      .split(/\n{2,}/)
      .map((p) => p.trim())
      .filter((p) => p.length > 0)
  }, [cachedSection?.content])

  return (
    <aside
      id="source-inspector-pane"
      aria-label="Grounded SEC Form 10-K Source Viewer"
      className="h-full flex flex-col surface-panel border-l border-hairline w-full"
    >
      {/* Pane Header & Mode Switcher */}
      <div className="px-4 py-2.5 border-b border-hairline flex items-center justify-between gap-2 shrink-0">
        <div className="flex items-center gap-1 p-0.5 rounded-lg surface-subtle border border-hairline">
          <button
            id="tab-grounded-citations"
            type="button"
            onClick={() => onChangeTab('citations')}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium cursor-pointer transition-colors ${
              activeTab === 'citations'
                ? 'surface-panel text-ink-primary shadow-xs'
                : 'text-ink-secondary hover:text-ink-primary'
            }`}
          >
            <Layers className="w-3.5 h-3.5" aria-hidden="true" />
            <span>Grounded Chunks</span>
            <span className="tabular-nums-mono text-[11px] px-1.5 py-0.2 rounded badge-accent">
              {citations.length}
            </span>
          </button>

          <button
            id="tab-full-10k-reader"
            type="button"
            onClick={() => {
              onChangeTab('document')
              if (!cachedSection && filingTargets[0]) {
                onRequestFilingSection(
                  filingTargets[0].ticker,
                  filingTargets[0].fiscalYear,
                  'item_7'
                )
              }
            }}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium cursor-pointer transition-colors ${
              activeTab === 'document'
                ? 'surface-panel text-ink-primary shadow-xs'
                : 'text-ink-secondary hover:text-ink-primary'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" aria-hidden="true" />
            <span>Full 10-K Filing</span>
          </button>
        </div>

        <button
          id="btn-close-source-pane-mobile"
          type="button"
          onClick={onCloseMobilePane}
          className="lg:hidden p-1 rounded text-ink-muted hover:text-ink-primary cursor-pointer"
          aria-label="Close Source Viewer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* TAB 1: Grounded Citations List */}
      {activeTab === 'citations' && (
        <div className="flex-1 flex flex-col min-h-0">
          <div className="px-4 py-2 border-b border-hairline flex items-center justify-between gap-2 text-xs shrink-0 surface-subtle">
            <span className="text-ink-muted text-[11px]">
              Hybrid Relevance Ranked (768-d + Lexical)
            </span>
            <div className="flex items-center gap-1">
              {(['ALL', 'item_7', 'item_1a'] as const).map((secKey) => (
                <button
                  key={secKey}
                  id={`btn-filter-cit-${secKey}`}
                  type="button"
                  onClick={() => setCitationSectionFilter(secKey)}
                  className={`px-2 py-0.5 rounded text-[11px] font-medium cursor-pointer transition-colors ${
                    citationSectionFilter === secKey
                      ? 'badge-accent'
                      : 'text-ink-muted hover:text-ink-primary'
                  }`}
                >
                  {secKey === 'ALL'
                    ? 'All'
                    : secKey === 'item_7'
                      ? 'Item 7'
                      : 'Item 1A'}
                </button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-3.5 space-y-2.5">
            {filteredCitations.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center px-6 py-12">
                <FileText
                  className="w-8 h-8 text-ink-muted mb-2.5 stroke-[1.5]"
                  aria-hidden="true"
                />
                <p className="text-sm font-medium text-ink-primary">
                  No 10-K passages selected yet
                </p>
                <p className="text-xs text-ink-muted mt-1 leading-relaxed max-w-xs">
                  Submit a natural language query or click any inline citation tag to
                  inspect grounded Form 10-K excerpts and full cached sections here.
                </p>
                {filingTargets[0] && (
                  <button
                    id="btn-browse-sample-10k"
                    type="button"
                    onClick={() => {
                      onChangeTab('document')
                      onRequestFilingSection(
                        filingTargets[0].ticker,
                        filingTargets[0].fiscalYear,
                        'item_7'
                      )
                    }}
                    className="mt-4 px-3 py-1.5 rounded-md text-xs font-medium surface-subtle border border-hairline text-ink-primary hover:border-strong cursor-pointer"
                  >
                    Browse {filingTargets[0].ticker} FY{filingTargets[0].fiscalYear}{' '}
                    Form 10-K
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
                    className={`rounded-lg p-3.5 border transition-all cursor-pointer ${
                      isSelected
                        ? 'surface-filing-sheet border-strong ring-1 ring-[var(--border-strong)]'
                        : 'surface-panel border-hairline hover:border-strong'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="tabular-nums-mono text-[11px] font-semibold text-ink-primary">
                          #{index + 1} · {cit.ticker} FY{cit.fiscal_year}
                        </span>
                        <span
                          className={`px-1.5 py-0.2 rounded text-[10px] font-semibold ${
                            isRisk ? 'badge-amber' : 'badge-accent'
                          }`}
                        >
                          {isRisk ? 'Item 1A · Risk' : 'Item 7 · MD&A'}
                        </span>
                      </div>

                      <span
                        className="tabular-nums-mono text-[11px] text-ink-muted"
                        title="Hybrid Dense + Lexical Retrieval Score"
                      >
                        Score {cit.score.toFixed(3)}
                      </span>
                    </div>

                    <div className="text-xs font-medium text-ink-secondary mb-1.5 truncate">
                      {cit.section_title}
                    </div>

                    <blockquote
                      className={`filing-prose text-[13.5px] leading-relaxed rounded p-2.5 ${
                        isSelected ? 'passage-highlight' : 'surface-subtle'
                      }`}
                    >
                      &ldquo;{clean10kExcerpt(cit.excerpt)}&rdquo;
                    </blockquote>

                    <div className="mt-2.5 pt-2 border-t border-hairline flex items-center justify-between gap-2 text-[11px]">
                      <code className="tabular-nums-mono text-[10px] text-ink-muted truncate">
                        {cit.chunk_id}
                      </code>

                      <div className="flex items-center gap-2.5 shrink-0">
                        <button
                          id={`btn-read-in-10k-${cit.chunk_id}`}
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            onSelectChunkId(cit.chunk_id)
                            onChangeTab('document')
                            onRequestFilingSection(
                              cit.ticker,
                              cit.fiscal_year,
                              cit.section_id,
                              cit.excerpt
                            )
                          }}
                          className="font-medium hover:underline cursor-pointer"
                          style={{ color: 'var(--accent-ink)' }}
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
                            className="inline-flex items-center gap-0.5 text-ink-muted hover:text-ink-primary"
                            title="Open official SEC EDGAR HTML archive"
                          >
                            <span>EDGAR</span>
                            <ExternalLink className="w-3 h-3" />
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
      )}

      {/* TAB 2: Full Cached 10-K Section Document Reader */}
      {activeTab === 'document' && (
        <div className="flex-1 flex flex-col min-h-0">
          {/* Filing & Section Selector Controls */}
          <div className="px-4 py-2.5 border-b border-hairline surface-subtle space-y-2 shrink-0">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-1.5">
                <label className="sr-only" htmlFor="select-doc-filing">
                  Select Issuer &amp; Fiscal Year
                </label>
                <select
                  id="select-doc-filing"
                  value={
                    cachedSection
                      ? `${cachedSection.ticker}_${cachedSection.fiscal_year}`
                      : filingTargets[0]
                        ? `${filingTargets[0].ticker}_${filingTargets[0].fiscalYear}`
                        : ''
                  }
                  onChange={(e) => {
                    const [t, fyStr] = e.target.value.split('_')
                    if (t && fyStr) {
                      onRequestFilingSection(
                        t,
                        Number(fyStr),
                        cachedSection?.section_id ?? 'item_7'
                      )
                    }
                  }}
                  className="px-2.5 py-1 rounded text-xs font-semibold tabular-nums-mono surface-panel border border-hairline text-ink-primary cursor-pointer"
                >
                  {filingTargets.map((ft) => (
                    <option
                      key={`${ft.ticker}_${ft.fiscalYear}`}
                      value={`${ft.ticker}_${ft.fiscalYear}`}
                    >
                      {ft.ticker} · FY{ft.fiscalYear} Form 10-K
                    </option>
                  ))}
                  {cachedSection &&
                    !filingTargets.some(
                      (ft) =>
                        ft.ticker === cachedSection.ticker &&
                        ft.fiscalYear === cachedSection.fiscal_year
                    ) && (
                      <option
                        value={`${cachedSection.ticker}_${cachedSection.fiscal_year}`}
                      >
                        {cachedSection.ticker} · FY{cachedSection.fiscal_year} Form
                        10-K
                      </option>
                    )}
                </select>

                <div className="inline-flex rounded surface-panel border border-hairline p-0.5">
                  {(['item_7', 'item_1a'] as const).map((secId) => {
                    const active = (cachedSection?.section_id ?? 'item_7') === secId
                    return (
                      <button
                        key={secId}
                        id={`btn-doc-section-${secId}`}
                        type="button"
                        onClick={() => {
                          const targetTicker =
                            cachedSection?.ticker ??
                            filingTargets[0]?.ticker ??
                            'NVDA'
                          const targetYear =
                            cachedSection?.fiscal_year ??
                            filingTargets[0]?.fiscalYear ??
                            2026
                          onRequestFilingSection(targetTicker, targetYear, secId)
                        }}
                        className={`px-2 py-0.5 rounded text-[11px] font-medium cursor-pointer transition-colors ${
                          active
                            ? 'badge-accent'
                            : 'text-ink-secondary hover:text-ink-primary'
                        }`}
                      >
                        {secId === 'item_7' ? 'Item 7 · MD&A' : 'Item 1A · Risk'}
                      </button>
                    )
                  })}
                </div>
              </div>

              {cachedSection?.sec_filing_url && (
                <a
                  id="link-doc-official-sec"
                  href={cachedSection.sec_filing_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-[11px] font-medium hover:underline"
                  style={{ color: 'var(--accent-ink)' }}
                >
                  <span>SEC.gov Archive</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>

            {/* In-Filing Keyword Search */}
            <div className="relative">
              <Search
                className="w-3.5 h-3.5 text-ink-muted absolute left-2.5 top-1/2 -translate-y-1/2"
                aria-hidden="true"
              />
              <input
                id="input-filing-keyword-search"
                type="text"
                value={docKeyword}
                onChange={(e) => setDocKeyword(e.target.value)}
                placeholder="Find term within cached 10-K section..."
                className="w-full pl-8 pr-7 py-1 rounded text-xs surface-panel border border-hairline text-ink-primary focus:outline-none focus:border-strong"
              />
              {docKeyword && (
                <button
                  id="btn-clear-filing-keyword"
                  type="button"
                  onClick={() => setDocKeyword('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-ink-muted hover:text-ink-primary cursor-pointer"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>

          {/* Document Body Sheet */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-5 surface-filing-sheet">
            {isLoadingSection ? (
              <div className="py-16 flex flex-col items-center justify-center gap-2 text-xs text-ink-muted">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Loading cached Form 10-K section from Context Cache...</span>
              </div>
            ) : cachedSection ? (
              <div className="max-w-prose mx-auto">
                {/* Official Filing Metadata Masthead */}
                <div className="pb-3.5 mb-4 border-b border-hairline">
                  <div className="flex items-center justify-between gap-2 flex-wrap mb-1">
                    <span className="text-[11px] uppercase tracking-wider font-semibold text-ink-muted tabular-nums-mono">
                      UNITED STATES SEC · FORM 10-K
                    </span>
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded badge-positive text-[10px] tabular-nums-mono font-medium">
                      <FileCheck2 className="w-3 h-3" />
                      Cache {cachedSection.cache_hit ? 'HIT' : 'LOADED'} · ~
                      {cachedSection.token_estimate.toLocaleString()} tok
                    </span>
                  </div>

                  <h3 className="text-base font-semibold text-ink-primary">
                    {cachedSection.company_name} ({cachedSection.ticker}) — FY
                    {cachedSection.fiscal_year}
                  </h3>

                  <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-muted tabular-nums-mono">
                    {cachedSection.cik && <span>CIK: {cachedSection.cik}</span>}
                    {cachedSection.accession_number && (
                      <span>Acc: {cachedSection.accession_number}</span>
                    )}
                    {cachedSection.period_of_report && (
                      <span>Period End: {cachedSection.period_of_report}</span>
                    )}
                    {cachedSection.filing_date && (
                      <span>Filed: {cachedSection.filing_date}</span>
                    )}
                  </div>
                </div>

                {/* Editorial Serif 10-K Section Prose */}
                <div className="filing-prose space-y-3.5">
                  {documentParagraphs.map((para, pIdx) => {
                    const isHeading =
                      para.startsWith('## ') ||
                      para.startsWith('### ') ||
                      (para.startsWith('**') && para.endsWith('**') && para.length < 140)

                    if (isHeading) {
                      const cleanTitle = para
                        .replace(/^#{2,4}\s*/, '')
                        .replace(/^\*\*|\*\*$/g, '')
                      return (
                        <h4
                          key={`doc-h-${pIdx}`}
                          className="font-sans text-xs font-bold uppercase tracking-wider text-ink-primary pt-3 pb-0.5"
                        >
                          {clean10kDocumentBlock(cleanTitle)}
                        </h4>
                      )
                    }

                    const spans = highlightTextSnippet(
                      para,
                      highlightedExcerpt,
                      docKeyword
                    )
                    const containsActiveExcerpt = spans.some((s) => s.active)

                    return (
                      <p
                        key={`doc-p-${pIdx}`}
                        ref={
                          containsActiveExcerpt
                            ? (el) => {
                                highlightedDocMarkRef.current = el
                              }
                            : undefined
                        }
                        className="leading-relaxed"
                      >
                        {spans.map((span, sIdx) =>
                          span.active ? (
                            <mark
                              key={`sp-${pIdx}-${sIdx}`}
                              className="passage-highlight-active"
                            >
                              {span.text}
                            </mark>
                          ) : span.matchedSearch ? (
                            <mark
                              key={`sp-${pIdx}-${sIdx}`}
                              className="passage-highlight"
                            >
                              {span.text}
                            </mark>
                          ) : (
                            <span key={`sp-${pIdx}-${sIdx}`}>{span.text}</span>
                          )
                        )}
                      </p>
                    )
                  })}
                </div>
              </div>
            ) : (
              <p className="text-xs text-ink-muted text-center py-12">
                Select an issuer and Form 10-K section above to inspect the filing.
              </p>
            )}
          </div>
        </div>
      )}
    </aside>
  )
}
