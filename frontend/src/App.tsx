import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AuthGatePanel } from './components/AuthGatePanel'
import { HeaderBar } from './components/HeaderBar'
import { ChatTurnItem } from './components/ChatTurnItem'
import { QueryComposer } from './components/QueryComposer'
import { SourceInspectorPane } from './components/SourceInspectorPane'
import {
  StarterPrompts,
  type StarterPromptPreset,
} from './components/StarterPrompts'
import {
  ApiAuthError,
  buildExternalOidcAuthorizeUrl,
  consumeOidcFragmentIdToken,
  exchangeBearerTokenSession,
  fetchAuthDiscoveryConfig,
  fetchAuthenticatedUser,
  getStoredBearerToken,
  loginWithWorkspaceOktaDevBroker,
  logoutAuthenticatedSession,
} from './services/authService'
import {
  executeConversationalRagQueryStream,
  fetchCached10kSection,
  fetchCompaniesCatalog,
  postAnalystFeedback,
} from './services/edgarApi'
import type {
  AuthDiscoveryConfig,
  AuthenticatedUserProfile,
  CachedSectionResponse,
  CatalogLoadStatus,
  CompanySummary,
  ConversationTurn,
  GroundedCitation,
  XbrlFactSelection,
} from './types/edgar'
import type {
  ParsedInlineCitationToken,
  ParsedInlineXbrlToken,
} from './utils/formatters'
import { getRenderableVarianceReports } from './utils/deterministicEvidence'
import { countDistinctCitations } from './utils/citationDedup'
import { filingHighlightAnchor } from './utils/filingMarkdown'
import { buildXbrlFactFromInlineToken } from './utils/xbrlFactSelection'

export function App() {
  // IAP + Google Workspace Okta/SSO Authentication State
  const [authConfig, setAuthConfig] = useState<AuthDiscoveryConfig | null>(null)
  const [authRequired, setAuthRequired] = useState<boolean>(false)
  const [authenticatedUser, setAuthenticatedUser] =
    useState<AuthenticatedUserProfile | null>(null)
  const [isAuthenticating, setIsAuthenticating] = useState<boolean>(false)
  const [authErrorMessage, setAuthErrorMessage] = useState<string | null>(null)
  const [isIapSession, setIsIapSession] = useState<boolean>(false)

  const [companies, setCompanies] = useState<CompanySummary[]>([])
  const [totalProfiles, setTotalProfiles] = useState<number | null>(null)
  const [catalogStatus, setCatalogStatus] = useState<CatalogLoadStatus>('loading')
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [yearBounds, setYearBounds] = useState<{ min: number; max: number }>({
    min: 2020,
    max: 2026,
  })

  const [promptInput, setPromptInput] = useState<string>('')
  const [turns, setTurns] = useState<ConversationTurn[]>([])

  // Split-pane 10-K Source Viewer state
  const [isSourcePaneOpen, setIsSourcePaneOpen] = useState<boolean>(true)
  const [sourceTab, setSourceTab] = useState<'citations' | 'document' | 'xbrl'>(
    'citations'
  )
  const [selectedXbrlFact, setSelectedXbrlFact] =
    useState<XbrlFactSelection | null>(null)
  const [activeCitations, setActiveCitations] = useState<GroundedCitation[]>([])
  const [activeChunkId, setActiveChunkId] = useState<string | null>(null)
  // Bumped on each inline-citation click so the source pane reveals the chunk
  // even when it is already the active one.
  const [chunkFocusNonce, setChunkFocusNonce] = useState<number>(0)
  const [cachedSection, setCachedSection] =
    useState<CachedSectionResponse | null>(null)
  const [isLoadingSection, setIsLoadingSection] = useState<boolean>(false)
  const [sectionError, setSectionError] = useState<string | null>(null)
  const [highlightedExcerpt, setHighlightedExcerpt] = useState<string | null>(
    null
  )

  const streamEndRef = useRef<HTMLDivElement | null>(null)

  const handleApiAuthFailure = useCallback((err: ApiAuthError) => {
    setAuthErrorMessage(err.message)
    if (err.status === 401) {
      setAuthenticatedUser(null)
      setAuthRequired(true)
    }
  }, [])

  const loadCompaniesCatalog = useCallback(async () => {
    setCatalogStatus('loading')
    setCatalogError(null)
    try {
      const catalog = await fetchCompaniesCatalog()
      setCompanies(catalog.companies)
      setTotalProfiles(catalog.total_profiles)
      setYearBounds({ min: catalog.min_year, max: catalog.max_year })
      setCatalogStatus('ready')
    } catch (err) {
      setCompanies([])
      setTotalProfiles(null)
      setCatalogStatus('error')
      if (err instanceof ApiAuthError) {
        setCatalogError(err.message)
        handleApiAuthFailure(err)
        return
      }
      setCatalogError(
        err instanceof Error
          ? err.message
          : 'Backend unreachable — could not load the company catalog. No data was shown.'
      )
    }
  }, [handleApiAuthFailure])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'terminal')
  }, [])

  // Discover IAP / Google Workspace Okta SSO configuration & initialize session
  useEffect(() => {
    let active = true

    async function bootstrapAuthAndCatalog() {
      const fragmentToken = consumeOidcFragmentIdToken()
      const { backendReachable, config } = await fetchAuthDiscoveryConfig()
      if (!active) return

      if (!backendReachable || !config) {
        await loadCompaniesCatalog()
        return
      }

      setAuthConfig(config)
      setAuthRequired(config.auth_required)

      if (fragmentToken) {
        setIsAuthenticating(true)
        try {
          const user = await exchangeBearerTokenSession(fragmentToken)
          if (!active) return
          setAuthenticatedUser(user)
          setAuthErrorMessage(null)
          await loadCompaniesCatalog()
        } catch (err) {
          if (!active) return
          setAuthErrorMessage(
            err instanceof Error
              ? err.message
              : 'Failed to validate Google Workspace / Okta SSO token.'
          )
        } finally {
          if (active) setIsAuthenticating(false)
        }
        return
      }

      if (getStoredBearerToken()) {
        try {
          const user = await fetchAuthenticatedUser()
          if (!active) return
          setAuthenticatedUser(user)
          setAuthErrorMessage(null)
          await loadCompaniesCatalog()
          return
        } catch (err) {
          if (!active) return
          if (err instanceof ApiAuthError) {
            setAuthErrorMessage(err.message)
          }
          setAuthenticatedUser(null)
        }
      }

      if (config.auth_required && config.iap?.session_auth_enabled) {
        // Behind Cloud IAP the browser already holds an IAP session, and IAP adds a
        // signed X-Goog-IAP-JWT-Assertion header to every request. Probe /auth/me
        // without a Bearer token; a 401 just means no IAP session, so stay quiet.
        try {
          const user = await fetchAuthenticatedUser()
          if (!active) return
          setAuthenticatedUser(user)
          setIsIapSession(true)
          setAuthErrorMessage(null)
          await loadCompaniesCatalog()
          return
        } catch {
          if (!active) return
        }
      }

      if (!config.auth_required) {
        await loadCompaniesCatalog()
      }
    }

    void bootstrapAuthAndCatalog()
    return () => {
      active = false
    }
  }, [loadCompaniesCatalog])

  const handleWorkspaceOktaSignIn = async () => {
    setAuthErrorMessage(null)

    const defaultDomain =
      authConfig?.identity_provider.allowed_workspace_domains[0] ||
      'fde-workspace.corp'
    const defaultGroups = authConfig?.identity_provider
      .required_analyst_groups?.length
      ? authConfig.identity_provider.required_analyst_groups
      : ['sec-edgar-analysts']
    const defaultEmail = `analyst@${defaultDomain}`

    if (authConfig) {
      const externalAuthorizeUrl = buildExternalOidcAuthorizeUrl(
        authConfig,
        defaultEmail
      )
      if (externalAuthorizeUrl) {
        window.location.assign(externalAuthorizeUrl)
        return
      }
    }

    setIsAuthenticating(true)
    try {
      const user = await loginWithWorkspaceOktaDevBroker({
        email: defaultEmail,
        name: 'Senior Financial Analyst',
        groups: defaultGroups,
      })
      setAuthenticatedUser(user)
      setAuthErrorMessage(null)
      await loadCompaniesCatalog()
    } catch (err) {
      setAuthErrorMessage(
        err instanceof Error
          ? err.message
          : 'Google Workspace / Okta SSO authentication failed.'
      )
    } finally {
      setIsAuthenticating(false)
    }
  }

  const handleBearerJwtSignIn = async (rawJwt: string) => {
    setAuthErrorMessage(null)
    setIsAuthenticating(true)
    try {
      const user = await exchangeBearerTokenSession(rawJwt)
      setAuthenticatedUser(user)
      setAuthErrorMessage(null)
      await loadCompaniesCatalog()
    } catch (err) {
      setAuthErrorMessage(
        err instanceof Error
          ? err.message
          : 'IAP Bearer JWT verification failed.'
      )
    } finally {
      setIsAuthenticating(false)
    }
  }

  const handleLogout = async () => {
    if (isIapSession) {
      // The IAP session cookie is what signs this user in, so clear it at IAP.
      window.location.assign('/?gcp-iap-mode=CLEAR_LOGIN_COOKIE')
      return
    }
    await logoutAuthenticatedSession()
    setAuthenticatedUser(null)
    setTurns([])
    setActiveCitations([])
    setActiveChunkId(null)
    setSelectedXbrlFact(null)
    setCachedSection(null)
    setSectionError(null)
    setHighlightedExcerpt(null)
    setAuthErrorMessage(null)
  }

  const distinctCitationCount = useMemo(
    () => countDistinctCitations(activeCitations),
    [activeCitations]
  )

  const isAuthLocked = authRequired && !authenticatedUser
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

  // Scope (companies, sector, section, fiscal years) is resolved by the backend
  // from the question text and conversation memory; the UI sends no filters.
  const runQuery = async (queryText: string) => {
    const trimmed = queryText.trim()
    if (!trimmed) return

    const turnId = `turn-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`

    const pendingTurn: ConversationTurn = {
      id: turnId,
      createdAt: new Date().toISOString(),
      userQuery: trimmed,
      response: null,
      isLoading: true,
    }

    setTurns((prev) => [...prev, pendingTurn])
    setPromptInput('')
    scrollToLatestTurn()

    try {
      const resp = await executeConversationalRagQueryStream(
        trimmed,
        (progress) => {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === turnId
                ? {
                    ...t,
                    progressStage: progress.stage,
                    progressMessage: progress.message,
                    progressMeta: progress.meta ?? null,
                    progressElapsedMs: progress.elapsed_ms,
                  }
                : t
            )
          )
        }
      )

      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId ? { ...t, response: resp, isLoading: false } : t
        )
      )

      const respCitations = resp.citations ?? []
      if (respCitations.length > 0) {
        setActiveCitations(respCitations)
        setActiveChunkId(null)
        setSourceTab('citations')
        setIsSourcePaneOpen(true)
      }
    } catch (err) {
      if (err instanceof ApiAuthError) {
        setTurns((prev) => prev.filter((t) => t.id !== turnId))
        handleApiAuthFailure(err)
        return
      }
      const message =
        err instanceof Error
          ? err.message
          : 'Backend unreachable — could not reach the EDGAR API. No data was shown.'
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
    void runQuery(preset.query)
  }

  const handleInspectXbrlFact = (fact: XbrlFactSelection) => {
    setSelectedXbrlFact(fact)
    setSourceTab('xbrl')
    setIsSourcePaneOpen(true)
  }

  const handleSelectInlineXbrlCitation = (
    token: ParsedInlineXbrlToken,
    turn: ConversationTurn
  ) => {
    // variance_reports is only populated when the agent invoked
    // calculate_period_variances; with no report, the fact is built from the
    // inline XBRL token alone.
    const reports = getRenderableVarianceReports(turn.response)
    const matchedRep = reports.find(
      (r) => r.ticker.toUpperCase() === token.ticker.toUpperCase()
    )
    handleInspectXbrlFact(buildXbrlFactFromInlineToken(token, matchedRep))
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
    setChunkFocusNonce((n) => n + 1)

    const matchedCit = turnCitations.find((c) => c.chunk_id === token.chunkId)
    if (sourceTab === 'document' && matchedCit) {
      void handleLoadFilingSection(
        matchedCit.ticker,
        matchedCit.fiscal_year,
        matchedCit.section_id,
        filingHighlightAnchor(matchedCit.excerpt)
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
    setSectionError(null)

    try {
      const sectionResp = await fetchCached10kSection(
        ticker,
        fiscalYear,
        sectionId
      )
      setCachedSection(sectionResp)
    } catch (err) {
      setCachedSection(null)
      setSectionError(
        err instanceof Error
          ? err.message
          : 'Backend unreachable — could not load the 10-K section. No data was shown.'
      )
      if (err instanceof ApiAuthError) {
        handleApiAuthFailure(err)
      }
    } finally {
      setIsLoadingSection(false)
    }
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
    postAnalystFeedback(
      turnId,
      rating,
      comment,
      targetTurn?.userQuery ?? ''
    ).catch((err) => {
      if (err instanceof ApiAuthError) {
        handleApiAuthFailure(err)
      }
    })
  }

  const handleResetConversation = () => {
    setTurns([])
    setActiveCitations([])
    setActiveChunkId(null)
    setSelectedXbrlFact(null)
    setHighlightedExcerpt(null)
    setPromptInput('')
  }

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col surface-canvas">
      <HeaderBar
        companyCount={catalogStatus === 'ready' ? companies.length : null}
        totalProfiles={totalProfiles}
        isBackendUnavailable={catalogStatus === 'error' && !isAuthLocked}
        turnCount={turns.length}
        onResetConversation={handleResetConversation}
        isSourcePaneOpen={isSourcePaneOpen}
        onToggleSourcePane={() => setIsSourcePaneOpen((prev) => !prev)}
        citationCount={distinctCitationCount}
        authenticatedUser={authenticatedUser}
        isAuthLocked={isAuthLocked}
        onLogout={() => {
          void handleLogout()
        }}
      />

      {isAuthLocked ? (
        <AuthGatePanel
          authConfig={authConfig}
          errorMessage={authErrorMessage}
          isAuthenticating={isAuthenticating}
          onSignInWorkspaceOkta={handleWorkspaceOktaSignIn}
          onSignInBearerJwt={handleBearerJwtSignIn}
        />
      ) : (
        <>
          {/* Split-Pane Main Terminal Workspace */}
          <div className="flex-1 flex min-h-0 relative overflow-hidden">
            {/* Left Blotter & Command Stream */}
            <main
              id="rag-conversation-pane"
              className="flex-1 flex flex-col min-w-0 min-h-0"
            >
              <div className="flex-1 overflow-y-auto px-3">
                {turns.length === 0 ? (
                  <StarterPrompts
                    companies={companies}
                    catalogStatus={catalogStatus}
                    catalogError={catalogError}
                    onSelectPreset={handleSelectStarter}
                    onSelectIssuerFiling={(t, fy) => {
                      void handleLoadFilingSection(t, fy, 'item_7')
                    }}
                  />
                ) : (
                  <div className="py-3">
                    {turns.map((turn) => (
                      <ChatTurnItem
                        key={turn.id}
                        turn={turn}
                        activeChunkId={activeChunkId}
                        onSelectInlineCitation={handleSelectInlineCitation}
                        onSelectXbrlCitation={handleSelectInlineXbrlCitation}
                        onInspectXbrlFact={handleInspectXbrlFact}
                        onSelectFollowup={(suggestedQuery) => {
                          void runQuery(suggestedQuery)
                        }}
                        onFocusTurnCitations={(cits) => {
                          setActiveCitations(cits)
                          setActiveChunkId(null)
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

              <QueryComposer
                value={promptInput}
                onChange={setPromptInput}
                onSubmit={() => {
                  void runQuery(promptInput)
                }}
                isSubmitting={isAnyTurnLoading}
                minYear={catalogStatus === 'ready' ? yearBounds.min : undefined}
                maxYear={catalogStatus === 'ready' ? yearBounds.max : undefined}
              />
            </main>

            {/* Right Split-Pane SEC Form 10-K Terminal Inspector */}
            {isSourcePaneOpen && (
              <div className="fixed inset-y-0 right-0 z-30 w-full sm:w-[440px] lg:static lg:z-auto lg:w-[44%] xl:w-[45%] shrink-0 h-full">
                <SourceInspectorPane
                  citations={activeCitations}
                  companies={companies}
                  activeChunkId={activeChunkId}
                  chunkFocusNonce={chunkFocusNonce}
                  onSelectChunkId={setActiveChunkId}
                  activeTab={sourceTab}
                  onChangeTab={setSourceTab}
                  selectedXbrlFact={selectedXbrlFact}
                  cachedSection={cachedSection}
                  isLoadingSection={isLoadingSection}
                  sectionError={sectionError}
                  onRequestFilingSection={(t, fy, sec, excerpt) => {
                    void handleLoadFilingSection(t, fy, sec, excerpt)
                  }}
                  highlightedExcerpt={highlightedExcerpt}
                  onCloseMobilePane={() => setIsSourcePaneOpen(false)}
                />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default App
