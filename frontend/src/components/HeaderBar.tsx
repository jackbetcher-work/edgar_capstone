import {
  Lock,
  LogOut,
  PanelRightClose,
  PanelRightOpen,
  RotateCcw,
} from 'lucide-react'
import type { AuthenticatedUserProfile } from '../types/edgar'

interface HeaderBarProps {
  companyCount: number
  totalProfiles: number
  turnCount: number
  onResetConversation: () => void
  isSourcePaneOpen: boolean
  onToggleSourcePane: () => void
  citationCount: number
  authenticatedUser?: AuthenticatedUserProfile | null
  isAuthLocked?: boolean
  onLogout?: () => void
}

export function HeaderBar({
  companyCount,
  totalProfiles,
  turnCount,
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
      className="surface-header border-b border-hairline px-3 h-9 flex items-center justify-between gap-3 shrink-0 z-20 select-none"
    >
      <div className="flex items-center gap-2.5 min-w-0 tabular-nums-mono">
        <span className="px-1.5 py-0.5 text-[11px] font-bold tracking-wider btn-terminal-go shrink-0">
          EDGR
        </span>
        <h1 className="text-xs font-semibold uppercase tracking-wider text-ink-primary truncate">
          EDGAR Analyst
        </h1>
        <span className="hidden sm:inline text-ink-muted">|</span>
        <span className="hidden sm:inline text-[11px] text-ink-secondary truncate">
          SEC 10-K TERMINAL
        </span>
        <span className="hidden md:inline text-ink-muted">|</span>
        <span className="hidden md:inline text-[11px] text-ink-muted truncate">
          {companyCount} ISSUERS · {totalProfiles} FILINGS
        </span>
      </div>

      <div className="flex items-center gap-1.5 shrink-0 tabular-nums-mono">
        <span
          id="terminal-session-status"
          className="inline-flex items-center gap-1.5 px-2 py-0.5 text-[10px] uppercase tracking-wider border border-hairline surface-subtle text-ink-secondary"
        >
          <span
            className="w-1.5 h-1.5 inline-block"
            style={{
              backgroundColor: isAuthLocked
                ? 'var(--bb-amber)'
                : 'var(--positive-ink)',
            }}
          />
          {isAuthLocked ? 'AUTH LOCKED' : 'READY'}
        </span>

        {authenticatedUser && (
          <div
            id="authenticated-user-pill"
            className="hidden lg:inline-flex items-center gap-1.5 px-2 py-0.5 surface-subtle border border-hairline text-[11px] text-ink-secondary"
            title={`Authenticated (${authenticatedUser.email})${
              authenticatedUser.cloud_identity_groups?.length
                ? ` · Cloud Identity: ${authenticatedUser.cloud_identity_groups.join(', ')}`
                : ''
            }`}
          >
            <Lock
              className="w-3 h-3 shrink-0"
              style={{ color: 'var(--positive-ink)' }}
              aria-hidden="true"
            />
            <span className="truncate max-w-[190px] text-ink-primary">
              {authenticatedUser.email}
            </span>
            {(() => {
              const primaryRole = authenticatedUser.roles.includes('admin')
                ? 'ADMIN'
                : authenticatedUser.roles.includes('analyst')
                  ? 'ANALYST'
                  : authenticatedUser.roles.includes('viewer')
                    ? 'VIEWER'
                    : null
              return primaryRole ? (
                <span className="px-1 py-0 text-[9px] font-semibold uppercase tracking-wider border border-hairline text-ink-muted">
                  {primaryRole}
                </span>
              ) : null
            })()}
          </div>
        )}

        {turnCount > 0 && !isAuthLocked && (
          <button
            id="btn-new-analysis-thread"
            type="button"
            onClick={onResetConversation}
            className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] uppercase tracking-wider btn-terminal-key cursor-pointer"
            title="Clear current terminal blotter"
          >
            <RotateCcw className="w-3 h-3" aria-hidden="true" />
            <span className="hidden sm:inline">Clear</span>
          </button>
        )}

        {!isAuthLocked && (
          <button
            id="btn-toggle-source-pane"
            type="button"
            onClick={onToggleSourcePane}
            className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-[11px] uppercase tracking-wider cursor-pointer ${
              isSourcePaneOpen
                ? 'btn-terminal-key-active'
                : 'btn-terminal-key'
            }`}
            aria-expanded={isSourcePaneOpen}
            title={
              isSourcePaneOpen
                ? 'Collapse 10-K Filing Pane'
                : 'Expand 10-K Filing Pane'
            }
          >
            {isSourcePaneOpen ? (
              <PanelRightClose className="w-3.5 h-3.5" aria-hidden="true" />
            ) : (
              <PanelRightOpen className="w-3.5 h-3.5" aria-hidden="true" />
            )}
            <span className="hidden sm:inline">10-K Pane</span>
            {citationCount > 0 && (
              <span className="text-[10px] font-bold">[{citationCount}]</span>
            )}
          </button>
        )}

        {authenticatedUser && onLogout && (
          <button
            id="btn-auth-logout"
            type="button"
            onClick={onLogout}
            className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] uppercase tracking-wider btn-terminal-key cursor-pointer"
            title="Sign out and revoke IAP Bearer session"
          >
            <LogOut className="w-3 h-3" aria-hidden="true" />
            <span className="hidden sm:inline">Sign Out</span>
          </button>
        )}
      </div>
    </header>
  )
}
