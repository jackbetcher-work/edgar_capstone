import { useEffect, useRef, useState } from 'react'
import { HeaderBar } from './components/HeaderBar'
import { HybridScopeBar } from './components/HybridScopeBar'
import { ChatTurnItem } from './components/ChatTurnItem'
import { QueryComposer } from './components/QueryComposer'
import { SourceInspectorPane } from './components/SourceInspectorPane'
import {
  StarterPrompts,
  type StarterPromptPreset,
} from './components/StarterPrompts'
import {
  executeConversationalRagQuery,
  fetchCached10kSection,
  fetchCompaniesCatalog,
  postAnalystFeedback,
} from './services/edgarApi'
import type {
  CachedSectionResponse,
  CompanySummary,
  ConversationTurn,
  GroundedCitation,
  SearchScopeFilters,
} from './types/edgar'
import type { ParsedInlineCitationToken } from './utils/formatters'

const DEFAULT_FILTERS: SearchScopeFilters = {
  selectedTickers: [],
  gicsSector: 'ALL',
  sectionId: 'ALL',
  startYear: 2024,
  endYear: 2026,
  topK: 8,
}

export function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const [isLiveBackend, setIsLiveBackend] = useState<boolean>(true)
  const [companies, setCompanies] = useState<CompanySummary[]>([])
  const [sectors, setSectors] = useState<string[]>([])
  const [totalProfiles, setTotalProfiles] = useState<number>(114)
  const [yearBounds, setYearBounds] = useState<{ min: number; max: number }>({
    min: 2020,
    max: 2026,
  })

  const [filters, setFilters] = useState<SearchScopeFilters>(DEFAULT_FILTERS)
  const [promptInput, setPromptInput] = useState<string>('')
  const [turns, setTurns] = useState<ConversationTurn[]>([])

  // Split-pane 10-K Source Viewer state
  const [isSourcePaneOpen, setIsSourcePaneOpen] = useState<boolean>(true)
  const [sourceTab, setSourceTab] = useState<'citations' | 'document'>(
    'citations'
  )
  const [activeCitations, setActiveCitations] = useState<GroundedCitation[]>([])
  const [activeChunkId, setActiveChunkId] = useState<string | null>(null)
  const [cachedSection, setCachedSection] =
    useState<CachedSectionResponse | null>(null)
  const [isLoadingSection, setIsLoadingSection] = useState<boolean>(false)
  const [highlightedExcerpt, setHighlightedExcerpt] = useState<string | null>(
    null
  )

  // Session telemetry
  const [cacheHits, setCacheHits] = useState<number>(0)
  const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null)

  const streamEndRef = useRef<HTMLDivElement | null>(null)

  // Apply light/dark data-theme attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  // Load curated S&P 500 catalog on startup
  useEffect(() => {
    let active = true
    fetchCompaniesCatalog().then(({ catalog, live }) => {
      if (!active) return
      setIsLiveBackend(live)
      setCompanies(catalog.companies)
      setSectors(catalog.gics_sectors)
      setTotalProfiles(catalog.total_profiles)
      setYearBounds({ min: catalog.min_year, max: catalog.max_year })
    })
    return () => {
      active = false
    }
  }, [])

  const isAnyTurnLoading = turns.some((t) => t.isLoading)

  const scrollToLatestTurn = () => {
    setTimeout(() => {
      if (
        streamEndRef.current &&
        typeof streamEndRef.current.scrollIntoView === 'function'
      ) {
        streamEndRef.current.scrollIntoView({
          behavior: 'smooth',
          block: 'end',
        })
      }
    }, 50)
  }

  const runQuery = async (queryText: string, scopeToUse: SearchScopeFilters) => {
    const trimmed = queryText.trim()
    if (!trimmed) return

    const turnId = `turn-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const snapshotFilters: SearchScopeFilters = {
      ...scopeToUse,
      selectedTickers: [...scopeToUse.selectedTickers],
    }

    const pendingTurn: ConversationTurn = {
      id: turnId,
      createdAt: new Date().toISOString(),
      userQuery: trimmed,
      appliedFilters: snapshotFilters,
      response: null,
      isLoading: true,
    }

    setTurns((prev) => [...prev, pendingTurn])
    setPromptInput('')
    scrollToLatestTurn()

    try {
      const resp = await executeConversationalRagQuery(trimmed, snapshotFilters)
      setIsLiveBackend(resp.connection_mode !== 'fallback')
      setLastLatencyMs(resp.total_latency_ms)
      if (resp.context_cache?.cache_hit) {
        setCacheHits((prev) => prev + 1)
      }

      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId ? { ...t, response: resp, isLoading: false } : t
        )
      )

      if (resp.citations.length > 0) {
        setActiveCitations(resp.citations)
        setActiveChunkId(resp.citations[0].chunk_id)
        setSourceTab('citations')
        setIsSourcePaneOpen(true)
      }
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Failed to complete grounded hybrid RAG query.'
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId
            ? { ...t, isLoading: false, errorMessage: message }
            : t
        )
      )
    }
  }

  const handleSelectStarter = (preset: StarterPromptPreset) => {
    const mergedScope: SearchScopeFilters = {
      ...filters,
      ...(preset.scopePatch ?? {}),
      selectedTickers: preset.scopePatch?.selectedTickers
        ? [...preset.scopePatch.selectedTickers]
        : [...filters.selectedTickers],
    }
    setFilters(mergedScope)
    void runQuery(preset.query, mergedScope)
  }

  const handleSelectInlineCitation = (
    token: ParsedInlineCitationToken,
    turnCitations: GroundedCitation[]
  ) => {
    setIsSourcePaneOpen(true)
    if (turnCitations.length > 0) {
      setActiveCitations(turnCitations)
    }
    setActiveChunkId(token.chunkId)

    const matchedCit = turnCitations.find((c) => c.chunk_id === token.chunkId)
    if (sourceTab === 'document' && matchedCit) {
      void handleLoadFilingSection(
        matchedCit.ticker,
        matchedCit.fiscal_year,
        matchedCit.section_id,
        matchedCit.excerpt
      )
    } else {
      setSourceTab('citations')
    }
  }

  const handleLoadFilingSection = async (
    ticker: string,
    fiscalYear: number,
    sectionId: 'item_7' | 'item_1a',
    excerptToHighlight?: string
  ) => {
    setIsSourcePaneOpen(true)
    setSourceTab('document')
    setHighlightedExcerpt(excerptToHighlight ?? null)
    setIsLoadingSection(true)

    const sectionResp = await fetchCached10kSection(
      ticker,
      fiscalYear,
      sectionId
    )
    setCachedSection(sectionResp)
    if (sectionResp.cache_hit) {
      setCacheHits((prev) => prev + 1)
    }
    setIsLoadingSection(false)
  }

  const handleFeedback = (
    turnId: string,
    rating: 'up' | 'down',
    comment: string
  ) => {
    const targetTurn = turns.find((t) => t.id === turnId)
    setTurns((prev) =>
      prev.map((t) =>
        t.id === turnId
          ? { ...t, feedbackRating: rating, feedbackComment: comment }
          : t
      )
    )
    void postAnalystFeedback(
      turnId,
      rating,
      comment,
      targetTurn?.userQuery ?? ''
    )
  }

  const handleResetConversation = () => {
    setTurns([])
    setActiveCitations([])
    setActiveChunkId(null)
    setHighlightedExcerpt(null)
    setPromptInput('')
  }

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col surface-canvas">
      <HeaderBar
        isLiveBackend={isLiveBackend}
        companyCount={companies.length || 18}
        totalProfiles={totalProfiles}
        turnCount={turns.length}
        cacheHits={cacheHits}
        lastLatencyMs={lastLatencyMs}
        theme={theme}
        onToggleTheme={() =>
          setTheme((prev) => (prev === 'light' ? 'dark' : 'light'))
        }
        onResetConversation={handleResetConversation}
        isSourcePaneOpen={isSourcePaneOpen}
        onToggleSourcePane={() => setIsSourcePaneOpen((prev) => !prev)}
        citationCount={activeCitations.length}
      />

      <HybridScopeBar
        companies={companies}
        sectors={sectors}
        minYear={yearBounds.min}
        maxYear={yearBounds.max}
        filters={filters}
        onChangeFilters={setFilters}
      />

      {/* Split-Pane Main Workspace */}
      <div className="flex-1 flex min-h-0 relative overflow-hidden">
        {/* Left/Center Conversational Agentic RAG Stream */}
        <main
          id="rag-conversation-pane"
          className="flex-1 flex flex-col min-w-0 min-h-0"
        >
          <div className="flex-1 overflow-y-auto px-4 sm:px-6">
            <div className="max-w-3xl mx-auto">
              {turns.length === 0 ? (
                <StarterPrompts onSelectPreset={handleSelectStarter} />
              ) : (
                <div className="py-2">
                  {turns.map((turn) => (
                    <ChatTurnItem
                      key={turn.id}
                      turn={turn}
                      activeChunkId={activeChunkId}
                      onSelectInlineCitation={handleSelectInlineCitation}
                      onFocusTurnCitations={(cits) => {
                        setActiveCitations(cits)
                        setActiveChunkId(cits[0]?.chunk_id ?? null)
                        setSourceTab('citations')
                        setIsSourcePaneOpen(true)
                      }}
                      onInspectFilingSection={(t, fy, sec) => {
                        void handleLoadFilingSection(t, fy, sec)
                      }}
                      onSubmitFeedback={handleFeedback}
                    />
                  ))}
                  <div ref={streamEndRef} />
                </div>
              )}
            </div>
          </div>

          <QueryComposer
            value={promptInput}
            onChange={setPromptInput}
            onSubmit={() => {
              void runQuery(promptInput, filters)
            }}
            isSubmitting={isAnyTurnLoading}
            filters={filters}
          />
        </main>

        {/* Right Split-Pane Grounded 10-K Source Viewer */}
        {isSourcePaneOpen && (
          <div className="fixed inset-y-0 right-0 z-30 w-full sm:w-[440px] lg:static lg:z-auto lg:w-[43%] xl:w-[44%] shrink-0 h-full shadow-2xl lg:shadow-none">
            <SourceInspectorPane
              citations={activeCitations}
              companies={companies}
              activeChunkId={activeChunkId}
              onSelectChunkId={setActiveChunkId}
              activeTab={sourceTab}
              onChangeTab={setSourceTab}
              cachedSection={cachedSection}
              isLoadingSection={isLoadingSection}
              onRequestFilingSection={(t, fy, sec, excerpt) => {
                void handleLoadFilingSection(t, fy, sec, excerpt)
              }}
              highlightedExcerpt={highlightedExcerpt}
              onCloseMobilePane={() => setIsSourcePaneOpen(false)}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default App
