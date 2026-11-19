import { useMemo, useState } from 'react'
import { Database, X } from 'lucide-react'
import type {
  CachedSectionResponse,
  CompanySummary,
  GroundedCitation,
  XbrlFactSelection,
} from '../types/edgar'
import {
  CitationsTab,
  type CitationSectionFilter,
  type FilingTarget,
} from './inspector/CitationsTab'
import { FilingReaderTab } from './inspector/FilingReaderTab'
import { XbrlFactTab } from './inspector/XbrlFactTab'
import { countDistinctCitations } from '../utils/citationDedup'

interface SourceInspectorPaneProps {
  citations: GroundedCitation[]
  companies: CompanySummary[]
  activeChunkId: string | null
  /** Incremented whenever an inline citation asks the pane to reveal a chunk. */
  chunkFocusNonce?: number
  onSelectChunkId: (chunkId: string | null) => void
  activeTab: 'citations' | 'document' | 'xbrl'
  onChangeTab: (tab: 'citations' | 'document' | 'xbrl') => void
  selectedXbrlFact?: XbrlFactSelection | null
  cachedSection: CachedSectionResponse | null
  isLoadingSection: boolean
  sectionError?: string | null
  onRequestFilingSection: (
    ticker: string,
    fiscalYear: number,
    sectionId: 'item_7' | 'item_1a',
    highlightExcerpt?: string
  ) => void
  highlightedExcerpt: string | null
  onCloseMobilePane: () => void
}

export function SourceInspectorPane({
  citations,
  companies,
  activeChunkId,
  chunkFocusNonce = 0,
  onSelectChunkId,
  activeTab,
  onChangeTab,
  selectedXbrlFact = null,
  cachedSection,
  isLoadingSection,
  sectionError = null,
  onRequestFilingSection,
  highlightedExcerpt,
  onCloseMobilePane,
}: SourceInspectorPaneProps) {
  const [citationSectionFilter, setCitationSectionFilter] =
    useState<CitationSectionFilter>('ALL')
  const [docKeyword, setDocKeyword] = useState('')
  const distinctCitationCount = useMemo(
    () => countDistinctCitations(citations),
    [citations]
  )

  const filingTargets = useMemo<FilingTarget[]>(() => {
    const map = new Map<string, FilingTarget>()
    citations.forEach((c) => {
      const key = `${c.ticker}_${c.fiscal_year}`
      if (!map.has(key)) {
        map.set(key, { ticker: c.ticker, fiscalYear: c.fiscal_year })
      }
    })
    if (map.size === 0) {
      companies.slice(0, 12).forEach((comp) => {
        map.set(`${comp.ticker}_${comp.latest_fiscal_year}`, {
          ticker: comp.ticker,
          fiscalYear: comp.latest_fiscal_year,
        })
      })
    }
    return Array.from(map.values())
  }, [citations, companies])

  const handleOpenFilingSection = (
    ticker: string,
    fiscalYear: number,
    sectionId: 'item_7' | 'item_1a',
    highlightExcerpt?: string
  ) => {
    onChangeTab('document')
    onRequestFilingSection(ticker, fiscalYear, sectionId, highlightExcerpt)
  }

  return (
    <aside
      id="source-inspector-pane"
      aria-label="Grounded SEC Form 10-K Source Viewer"
      className="h-full flex flex-col surface-panel border-l border-hairline w-full"
    >
      {/* Terminal Pane Tab Bar */}
      <div className="px-3 h-9 surface-header border-b border-hairline flex items-center justify-between gap-2 shrink-0 tabular-nums-mono">
        <div className="flex items-center gap-1">
          <button
            id="tab-grounded-citations"
            type="button"
            onClick={() => onChangeTab('citations')}
            className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-[11px] uppercase tracking-wider cursor-pointer ${
              activeTab === 'citations'
                ? 'btn-terminal-key-active font-semibold'
                : 'btn-terminal-key'
            }`}
          >
            <span>Grounded Chunks</span>
            <span className="text-[10px]">[{distinctCitationCount}]</span>
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
            className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-[11px] uppercase tracking-wider cursor-pointer ${
              activeTab === 'document'
                ? 'btn-terminal-key-active font-semibold'
                : 'btn-terminal-key'
            }`}
          >
            <span>Full 10-K Filing</span>
          </button>

          {selectedXbrlFact && (
            <button
              id="tab-xbrl-fact-provenance"
              type="button"
              onClick={() => onChangeTab('xbrl')}
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-[11px] uppercase tracking-wider cursor-pointer ${
                activeTab === 'xbrl'
                  ? 'btn-terminal-key-active font-semibold'
                  : 'btn-terminal-key'
              }`}
            >
              <Database className="w-3 h-3" />
              <span>XBRL Fact</span>
            </button>
          )}
        </div>

        <button
          id="btn-close-source-pane-mobile"
          type="button"
          onClick={onCloseMobilePane}
          className="lg:hidden p-1 text-ink-muted hover:text-ink-primary cursor-pointer"
          aria-label="Close Source Viewer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {activeTab === 'citations' && (
        <CitationsTab
          citations={citations}
          activeChunkId={activeChunkId}
          focusNonce={chunkFocusNonce}
          sectionFilter={citationSectionFilter}
          onChangeSectionFilter={setCitationSectionFilter}
          onSelectChunkId={onSelectChunkId}
          filingTargets={filingTargets}
          onOpenFilingSection={handleOpenFilingSection}
        />
      )}

      {activeTab === 'document' && (
        <FilingReaderTab
          filingTargets={filingTargets}
          cachedSection={cachedSection}
          isLoadingSection={isLoadingSection}
          sectionError={sectionError}
          highlightedExcerpt={highlightedExcerpt}
          keyword={docKeyword}
          onChangeKeyword={setDocKeyword}
          onRequestFilingSection={onRequestFilingSection}
        />
      )}

      {activeTab === 'xbrl' && (
        <XbrlFactTab
          selectedXbrlFact={selectedXbrlFact}
          onOpenFilingSection={handleOpenFilingSection}
        />
      )}
    </aside>
  )
}
