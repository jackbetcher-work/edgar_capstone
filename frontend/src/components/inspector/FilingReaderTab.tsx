import { useEffect, useMemo, useRef } from 'react'
import { AlertTriangle, ExternalLink, Loader2, Search, X } from 'lucide-react'
import type { CachedSectionResponse } from '../../types/edgar'
import { clean10kDocumentBlock, clean10kExcerpt } from '../../utils/formatters'
import type { FilingTarget } from './CitationsTab'

interface FilingReaderTabProps {
  filingTargets: FilingTarget[]
  cachedSection: CachedSectionResponse | null
  isLoadingSection: boolean
  sectionError: string | null
  highlightedExcerpt: string | null
  keyword: string
  onChangeKeyword: (value: string) => void
  onRequestFilingSection: (
    ticker: string,
    fiscalYear: number,
    sectionId: 'item_7' | 'item_1a',
    highlightExcerpt?: string
  ) => void
}

function highlightTextSnippet(
  text: string,
  targetSnippet: string | null,
  searchKeyword: string
): Array<{ text: string; active: boolean; matchedSearch: boolean }> {
  const cleaned = clean10kDocumentBlock(text)
  const cleanTarget = targetSnippet
    ? clean10kExcerpt(targetSnippet).slice(0, 95)
    : ''
  const cleanKeyword = searchKeyword.trim()

  if (cleanTarget && cleaned.toLowerCase().includes(cleanTarget.toLowerCase())) {
    const idx = cleaned.toLowerCase().indexOf(cleanTarget.toLowerCase())
    const matchLen = Math.min(
      cleaned.length - idx,
      Math.max(
        cleanTarget.length,
        targetSnippet ? clean10kExcerpt(targetSnippet).length : 0
      )
    )
    return [
      { text: cleaned.slice(0, idx), active: false, matchedSearch: false },
      {
        text: cleaned.slice(idx, idx + matchLen),
        active: true,
        matchedSearch: false,
      },
      {
        text: cleaned.slice(idx + matchLen),
        active: false,
        matchedSearch: false,
      },
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

export function FilingReaderTab({
  filingTargets,
  cachedSection,
  isLoadingSection,
  sectionError,
  highlightedExcerpt,
  keyword,
  onChangeKeyword,
  onRequestFilingSection,
}: FilingReaderTabProps) {
  const highlightedDocMarkRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (highlightedExcerpt && !isLoadingSection) {
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
  }, [highlightedExcerpt, isLoadingSection, cachedSection?.cache_key])

  const documentParagraphs = useMemo(() => {
    if (!cachedSection?.content) return []
    return cachedSection.content
      .split(/\n{2,}/)
      .map((p) => p.trim())
      .filter((p) => p.length > 0)
  }, [cachedSection?.content])

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="px-3 py-2 border-b border-hairline surface-subtle space-y-1.5 shrink-0 tabular-nums-mono">
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
              className="px-2 py-0.5 text-[11px] font-semibold surface-input border border-hairline text-bb-yellow cursor-pointer focus:outline-none focus:border-[var(--bb-amber)]"
            >
              {filingTargets.map((ft) => (
                <option
                  key={`${ft.ticker}_${ft.fiscalYear}`}
                  value={`${ft.ticker}_${ft.fiscalYear}`}
                >
                  {ft.ticker} · FY{ft.fiscalYear} 10-K
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
                    {cachedSection.ticker} · FY{cachedSection.fiscal_year} 10-K
                  </option>
                )}
            </select>

            <div className="inline-flex border border-hairline surface-input">
              {(['item_7', 'item_1a'] as const).map((secId) => {
                const active = (cachedSection?.section_id ?? 'item_7') === secId
                return (
                  <button
                    key={secId}
                    id={`btn-doc-section-${secId}`}
                    type="button"
                    onClick={() => {
                      const targetTicker =
                        cachedSection?.ticker ?? filingTargets[0]?.ticker
                      const targetYear =
                        cachedSection?.fiscal_year ??
                        filingTargets[0]?.fiscalYear
                      if (!targetTicker || targetYear === undefined) return
                      onRequestFilingSection(targetTicker, targetYear, secId)
                    }}
                    className={`px-2 py-0.5 text-[10px] uppercase font-semibold cursor-pointer transition-colors ${
                      active
                        ? 'badge-amber'
                        : 'text-ink-secondary hover:text-ink-primary'
                    }`}
                  >
                    {secId === 'item_7' ? 'ITEM 7 MD&A' : 'ITEM 1A RISK'}
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
              className="inline-flex items-center gap-1 text-[10px] uppercase font-semibold text-bb-cyan hover:underline"
            >
              <span>SEC.GOV</span>
              <ExternalLink className="w-2.5 h-2.5" />
            </a>
          )}
        </div>

        <div className="relative">
          <Search
            className="w-3 h-3 text-ink-muted absolute left-2 top-1/2 -translate-y-1/2"
            aria-hidden="true"
          />
          <input
            id="input-filing-keyword-search"
            type="text"
            value={keyword}
            onChange={(e) => onChangeKeyword(e.target.value)}
            placeholder="FIND TERM IN 10-K SECTION..."
            className="w-full pl-7 pr-6 py-0.5 text-[11px] surface-input border border-hairline text-ink-primary focus:outline-none focus:border-[var(--bb-amber)]"
          />
          {keyword && (
            <button
              id="btn-clear-filing-keyword"
              type="button"
              onClick={() => onChangeKeyword('')}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 text-ink-muted hover:text-ink-primary cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3.5 surface-filing-sheet">
        {isLoadingSection ? (
          <div className="py-14 flex flex-col items-center justify-center gap-2 text-xs text-bb-amber tabular-nums-mono">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>LOADING SEC FORM 10-K SECTION...</span>
          </div>
        ) : sectionError ? (
          <div
            id="section-load-error"
            role="alert"
            className="p-2.5 badge-negative text-xs flex items-start gap-2 tabular-nums-mono"
          >
            <AlertTriangle
              className="w-3.5 h-3.5 shrink-0 mt-0.5"
              aria-hidden="true"
            />
            <span>{sectionError}</span>
          </div>
        ) : cachedSection ? (
          <div>
            {/* Official Filing Metadata Masthead */}
            <div className="pb-2.5 mb-3 border-b border-hairline tabular-nums-mono">
              <div className="flex items-center justify-between gap-2 flex-wrap mb-0.5">
                <span className="text-[10px] uppercase tracking-wider font-bold text-bb-amber">
                  UNITED STATES SEC · FORM 10-K
                </span>
                <span className="text-[10px] uppercase text-ink-muted">
                  {cachedSection.section_id === 'item_1a'
                    ? 'ITEM 1A · RISK FACTORS'
                    : 'ITEM 7 · MD&A'}
                </span>
              </div>

              <h3 className="text-sm font-semibold text-ink-primary">
                {cachedSection.company_name} (
                <span className="text-bb-yellow">{cachedSection.ticker}</span>)
                — FY{cachedSection.fiscal_year}
              </h3>

              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-ink-muted">
                {cachedSection.cik && <span>CIK: {cachedSection.cik}</span>}
                {cachedSection.accession_number && (
                  <span>ACC: {cachedSection.accession_number}</span>
                )}
                {cachedSection.period_of_report && (
                  <span>PERIOD: {cachedSection.period_of_report}</span>
                )}
                {cachedSection.filing_date && (
                  <span>FILED: {cachedSection.filing_date}</span>
                )}
              </div>
            </div>

            {/* Terminal 10-K Section Prose */}
            <div className="filing-prose space-y-3">
              {documentParagraphs.map((para, pIdx) => {
                const isHeading =
                  para.startsWith('## ') ||
                  para.startsWith('### ') ||
                  (para.startsWith('**') &&
                    para.endsWith('**') &&
                    para.length < 140)

                if (isHeading) {
                  const cleanTitle = para
                    .replace(/^#{2,4}\s*/, '')
                    .replace(/^\*\*|\*\*$/g, '')
                  return (
                    <h4
                      key={`doc-h-${pIdx}`}
                      className="tabular-nums-mono text-[11px] font-bold uppercase tracking-wider text-bb-amber pt-2 pb-0.5 border-b border-hairline"
                    >
                      {clean10kDocumentBlock(cleanTitle)}
                    </h4>
                  )
                }

                const spans = highlightTextSnippet(
                  para,
                  highlightedExcerpt,
                  keyword
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
          <p className="text-xs text-ink-muted text-center py-12 tabular-nums-mono uppercase">
            SELECT AN EQUITY AND 10-K SECTION TO INSPECT FILING.
          </p>
        )}
      </div>
    </div>
  )
}
