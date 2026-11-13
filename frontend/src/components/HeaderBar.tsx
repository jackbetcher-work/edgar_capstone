import {
  BookOpenText,
  Database,
  Lock,
  LogOut,
  Moon,
  PanelRightClose,
  PanelRightOpen,
  RotateCcw,
  ShieldCheck,
  Sun,
} from 'lucide-react'
import type { AuthenticatedUserProfile } from '../types/edgar'

interface HeaderBarProps {
  isLiveBackend: boolean
  companyCount: number
  totalProfiles: number
  turnCount: number
  cacheHits: number
  lastLatencyMs: number | null
  theme: 'light' | 'dark'
  onToggleTheme: () => void
  onResetConversation: () => void
  isSourcePaneOpen: boolean
  onToggleSourcePane: () => void
  citationCount: number
  authenticatedUser?: AuthenticatedUserProfile | null
  isAuthLocked?: boolean
  onLogout?: () => void
}

export function HeaderBar({
  isLiveBackend,
  companyCount,
  totalProfiles,
  turnCount,
  cacheHits,
  lastLatencyMs,
  theme,
  onToggleTheme,
  onResetConversation,
  isSourcePaneOpen,
  onToggleSourcePane,
  citationCount,
  authenticatedUser = null,
  isAuthLocked = false,
  onLogout,
}: HeaderBarProps) {
  return (
    <header
      id="workspace-header"
      className="surface-panel border-b border-hairline px-4 sm:px-6 py-2.5 flex items-center justify-between gap-3 shrink-0 z-20"
    >
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-8 h-8 rounded-md btn-primary-ink flex items-center justify-center shrink-0">
          <BookOpenText className="w-4 h-4" aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2.5 flex-wrap">
            <h1 className="text-sm sm:text-[15px] font-semibold tracking-tight text-ink-primary truncate">
              EDGAR Analyst
            </h1>
            <span className="hidden md:inline-block text-xs text-ink-muted">·</span>
            <span className="hidden md:inline-block text-xs text-ink-secondary truncate">
              Source-Grounded SEC Form 10-K Hybrid RAG
            </span>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-ink-muted tabular-nums-mono mt-0.5">
            <span className="inline-flex items-center gap-1">
              <span
                className="w-1.5 h-1.5 rounded-full inline-block"
                style={{
                  backgroundColor: isAuthLocked
                    ? 'var(--amber-ink)'
                    : isLiveBackend
                      ? 'var(--positive-ink)'
                      : 'var(--amber-ink)',
                }}
              />
              {isAuthLocked
                ? 'IAP Protected · Sign-In Required'
                : isLiveBackend
                  ? 'Live FastAPI · RootOrchestrator'
                  : 'Local Corpus Snapshot'}
            </span>
            <span className="hidden sm:inline">·</span>
            <span className="hidden sm:inline-flex items-center gap-1">
              <Database className="w-3 h-3" aria-hidden="true" />
              {companyCount} S&amp;P 500 Issuers ({totalProfiles} Filings)
            </span>
            <span className="hidden lg:inline">·</span>
            <span className="hidden lg:inline-flex items-center gap-1">
              <ShieldCheck className="w-3 h-3" aria-hidden="true" />
              Strict Grounding Lock
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {authenticatedUser && (
          <div
            id="authenticated-user-pill"
            className="hidden md:inline-flex items-center gap-2 px-2.5 py-1 rounded-md surface-subtle border border-hairline text-[11px] text-ink-secondary tabular-nums-mono"
            title={`Authenticated via Google Workspace / Okta SSO (${authenticatedUser.identity_provider})`}
          >
            <Lock
              className="w-3 h-3"
              style={{ color: 'var(--positive-ink)' }}
              aria-hidden="true"
            />
            <span className="badge-positive px-1.5 py-0.2 rounded text-[10px] font-semibold">
              IAP · JWT
            </span>
            <span className="truncate max-w-[180px] text-ink-primary font-medium">
              {authenticatedUser.email}
            </span>
          </div>
        )}

        {turnCount > 0 && !isAuthLocked && (
          <div
            id="session-telemetry-pill"
            className="hidden xl:flex items-center gap-2.5 px-2.5 py-1 rounded-md surface-subtle border border-hairline text-[11px] text-ink-secondary tabular-nums-mono"
            title="Session Context Cache & Latency Telemetry"
          >
            <span>Cache Hits: {cacheHits}</span>
            {lastLatencyMs !== null && (
              <>
                <span className="text-ink-muted">|</span>
                <span>TTFT: {lastLatencyMs.toFixed(0)} ms</span>
              </>
            )}
          </div>
        )}

        {turnCount > 0 && !isAuthLocked && (
          <button
            id="btn-new-analysis-thread"
            type="button"
            onClick={onResetConversation}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium surface-subtle border border-hairline text-ink-secondary hover:text-ink-primary cursor-pointer transition-colors"
            title="Start a clean research thread"
          >
            <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
            <span className="hidden sm:inline">New Thread</span>
          </button>
        )}

        <button
          id="btn-toggle-theme"
          type="button"
          onClick={onToggleTheme}
          className="p-1.5 rounded-md surface-subtle border border-hairline text-ink-secondary hover:text-ink-primary cursor-pointer transition-colors"
          aria-label={
            theme === 'light'
              ? 'Switch to dark desk theme'
              : 'Switch to warm paper theme'
          }
          title={
            theme === 'light'
              ? 'Switch to dark desk theme'
              : 'Switch to warm paper theme'
          }
        >
          {theme === 'light' ? (
            <Moon className="w-4 h-4" aria-hidden="true" />
          ) : (
            <Sun className="w-4 h-4" aria-hidden="true" />
          )}
        </button>

        {!isAuthLocked && (
          <button
            id="btn-toggle-source-pane"
            type="button"
            onClick={onToggleSourcePane}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium surface-subtle border border-hairline text-ink-primary cursor-pointer transition-colors"
            aria-expanded={isSourcePaneOpen}
            title={
              isSourcePaneOpen
                ? 'Collapse 10-K Source Viewer'
                : 'Expand 10-K Source Viewer'
            }
          >
            {isSourcePaneOpen ? (
              <PanelRightClose className="w-4 h-4" aria-hidden="true" />
            ) : (
              <PanelRightOpen className="w-4 h-4" aria-hidden="true" />
            )}
            <span className="hidden sm:inline">10-K Sources</span>
            {citationCount > 0 && (
              <span className="badge-accent px-1.5 py-0.2 rounded text-[11px] tabular-nums-mono font-semibold">
                {citationCount}
              </span>
            )}
          </button>
        )}

        {authenticatedUser && onLogout && (
          <button
            id="btn-auth-logout"
            type="button"
            onClick={onLogout}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium surface-subtle border border-hairline text-ink-secondary hover:text-ink-primary cursor-pointer transition-colors"
            title="Sign out and revoke IAP Bearer session"
          >
            <LogOut className="w-3.5 h-3.5" aria-hidden="true" />
            <span className="hidden sm:inline">Sign Out</span>
          </button>
        )}
      </div>
    </header>
  )
}
