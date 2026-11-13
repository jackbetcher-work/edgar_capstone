import { useState } from 'react'
import {
  AlertCircle,
  ChevronDown,
  ChevronUp,
  KeyRound,
  Lock,
} from 'lucide-react'
import type { AuthDiscoveryConfig } from '../types/edgar'

interface AuthGatePanelProps {
  authConfig: AuthDiscoveryConfig | null
  errorMessage: string | null
  isAuthenticating: boolean
  onSignInWorkspaceOkta: () => Promise<void>
  onSignInBearerJwt: (rawJwt: string) => Promise<void>
}

export function AuthGatePanel({
  authConfig,
  errorMessage,
  isAuthenticating,
  onSignInWorkspaceOkta,
  onSignInBearerJwt,
}: AuthGatePanelProps) {
  const [showBearerPanel, setShowBearerPanel] = useState<boolean>(false)
  const [bearerJwtInput, setBearerJwtInput] = useState<string>('')

  const devBrokerEnabled =
    authConfig?.identity_provider.local_sso_dev_broker_enabled ?? false
  const hasExternalOidc = Boolean(
    authConfig?.identity_provider.authorization_endpoint &&
      (authConfig?.identity_provider.okta_client_id ||
        authConfig?.identity_provider.google_client_id)
  )
  const canSignInViaSso = hasExternalOidc || devBrokerEnabled

  const handlePrimarySsoSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    await onSignInWorkspaceOkta()
  }

  const handleBearerJwtSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!bearerJwtInput.trim()) return
    await onSignInBearerJwt(bearerJwtInput.trim())
  }

  return (
    <section
      id="iap-auth-gate"
      aria-labelledby="iap-auth-heading"
      className="flex-1 flex items-center justify-center p-4 overflow-y-auto"
    >
      <div className="w-full max-w-md surface-panel border border-hairline">
        <div className="px-4 py-2 surface-header border-b border-hairline flex items-center justify-between gap-2 tabular-nums-mono">
          <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-bb-amber">
            <Lock className="w-3 h-3" aria-hidden="true" />
            IAP · SSO TERMINAL ACCESS
          </span>
          <span className="text-[10px] text-ink-muted uppercase">
            SEC EDGAR 10-K
          </span>
        </div>

        <div className="p-5">
          <h2
            id="iap-auth-heading"
            className="text-base font-semibold text-ink-primary tabular-nums-mono"
          >
            Sign in with Google Workspace
          </h2>
          <p className="text-xs text-ink-secondary mt-1.5 leading-relaxed">
            Federated via Okta SSO. Verified IAP JWT Bearer session required to
            query audited XBRL variances and SEC Form 10-K filings.
          </p>

          {errorMessage && (
            <div
              id="auth-error-banner"
              role="alert"
              className="mt-4 p-2.5 badge-negative flex items-start gap-2 text-xs leading-relaxed tabular-nums-mono"
            >
              <AlertCircle
                className="w-3.5 h-3.5 shrink-0 mt-0.5"
                aria-hidden="true"
              />
              <span>{errorMessage}</span>
            </div>
          )}

          {canSignInViaSso && (
            <form onSubmit={handlePrimarySsoSubmit} className="mt-4">
              <button
                id="btn-sso-workspace-login"
                type="submit"
                disabled={isAuthenticating}
                className="w-full py-2 px-3 btn-terminal-go text-xs inline-flex items-center justify-between gap-2 cursor-pointer disabled:opacity-50"
              >
                <span>
                  {isAuthenticating
                    ? 'Verifying SSO Session...'
                    : 'Continue with Google Workspace (Okta SSO)'}
                </span>
                <span className="text-[11px]">&lt;GO&gt;</span>
              </button>
            </form>
          )}

          <div className="mt-4 pt-3.5 border-t border-hairline">
            <button
              id="btn-toggle-bearer-jwt"
              type="button"
              onClick={() => setShowBearerPanel((prev) => !prev)}
              aria-expanded={showBearerPanel || !canSignInViaSso}
              className="w-full flex items-center justify-between text-[11px] uppercase tracking-wider text-ink-secondary hover:text-ink-primary cursor-pointer tabular-nums-mono"
            >
              <span className="inline-flex items-center gap-1.5">
                <KeyRound className="w-3 h-3" aria-hidden="true" />
                Authenticate with IAP / OIDC Bearer JWT Token
              </span>
              {showBearerPanel || !canSignInViaSso ? (
                <ChevronUp className="w-3.5 h-3.5" aria-hidden="true" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" aria-hidden="true" />
              )}
            </button>

            {(showBearerPanel || !canSignInViaSso) && (
              <form onSubmit={handleBearerJwtSubmit} className="mt-2.5 space-y-2">
                <label
                  htmlFor="input-iap-bearer-jwt"
                  className="block text-[11px] text-ink-muted tabular-nums-mono"
                >
                  Paste a signed Google Cloud IAP, Workspace OIDC, or Okta Bearer
                  JWT:
                </label>
                <textarea
                  id="input-iap-bearer-jwt"
                  rows={3}
                  value={bearerJwtInput}
                  onChange={(e) => setBearerJwtInput(e.target.value)}
                  placeholder="eyJhbGciOiJFUzI1NiIsImtpZCI6..."
                  className="w-full px-2.5 py-1.5 surface-input border border-strong text-xs text-ink-primary tabular-nums-mono resize-none focus:outline-none focus:border-[var(--bb-amber)]"
                />
                <button
                  id="btn-submit-bearer-jwt"
                  type="submit"
                  disabled={isAuthenticating || !bearerJwtInput.trim()}
                  className="w-full py-1.5 px-3 btn-terminal-key text-xs font-semibold uppercase tracking-wider text-ink-primary cursor-pointer disabled:opacity-40"
                >
                  Verify &amp; Connect IAP Bearer JWT
                </button>
              </form>
            )}
          </div>
        </div>

        <div className="px-4 py-2 surface-subtle border-t border-hairline flex items-center justify-between gap-2 text-[10px] text-ink-muted tabular-nums-mono uppercase">
          <span>IDP: WORKSPACE / OKTA SSO</span>
          <span>IAP JWT (ES256 / RS256)</span>
        </div>
      </div>
    </section>
  )
}
