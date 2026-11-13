import { useState } from 'react'
import {
  AlertCircle,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  KeyRound,
  Lock,
  ShieldCheck,
  UserCheck,
} from 'lucide-react'
import type { AuthDiscoveryConfig } from '../types/edgar'

interface AuthGatePanelProps {
  authConfig: AuthDiscoveryConfig | null
  errorMessage: string | null
  isAuthenticating: boolean
  onSignInWorkspaceOkta: (params: {
    email: string
    name: string
    groups: string[]
  }) => Promise<void>
  onSignInBearerJwt: (rawJwt: string) => Promise<void>
}

export function AuthGatePanel({
  authConfig,
  errorMessage,
  isAuthenticating,
  onSignInWorkspaceOkta,
  onSignInBearerJwt,
}: AuthGatePanelProps) {
  const defaultDomain =
    authConfig?.identity_provider.allowed_workspace_domains[0] ||
    'fde-workspace.corp'
  const defaultGroup =
    authConfig?.identity_provider.required_analyst_groups[0] ||
    'sec-edgar-analysts'

  const [email, setEmail] = useState<string>(`analyst@${defaultDomain}`)
  const [displayName, setDisplayName] = useState<string>(
    'Senior Financial Analyst'
  )
  const [groupInput, setGroupInput] = useState<string>(defaultGroup)
  const [showSandboxClaims, setShowSandboxClaims] = useState<boolean>(false)
  const [showBearerPanel, setShowBearerPanel] = useState<boolean>(false)
  const [bearerJwtInput, setBearerJwtInput] = useState<string>('')

  const devBrokerEnabled =
    authConfig?.identity_provider.local_sso_dev_broker_enabled ?? false
  const hasExternalOidc = Boolean(
    authConfig?.identity_provider.authorization_endpoint &&
      (authConfig?.identity_provider.okta_client_id ||
        authConfig?.identity_provider.google_client_id)
  )

  const handlePrimarySsoSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const parsedGroups = groupInput
      .split(',')
      .map((g) => g.trim())
      .filter(Boolean)
    await onSignInWorkspaceOkta({
      email: email.trim(),
      name: displayName.trim(),
      groups: parsedGroups,
    })
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
      className="flex-1 flex items-center justify-center p-4 sm:p-8 overflow-y-auto"
    >
      <div className="w-full max-w-lg surface-panel border border-hairline rounded-xl p-6 sm:p-8 animate-turn-enter">
        <div className="flex items-start justify-between gap-4 pb-5 border-b border-hairline">
          <div className="space-y-1.5">
            <div className="inline-flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-ink-muted tabular-nums-mono">
              <Lock className="w-3.5 h-3.5" aria-hidden="true" />
              Identity-Aware Proxy · Internal API Perimeter
            </div>
            <h2
              id="iap-auth-heading"
              className="text-lg sm:text-xl font-semibold tracking-tight text-ink-primary"
            >
              Sign in with Google Workspace
            </h2>
            <p className="text-xs sm:text-sm text-ink-secondary leading-relaxed">
              Federated via <strong>Okta SSO</strong>. Internal SEC Form 10-K
              variance, RAG retrieval, and source-inspection APIs require
              verified <strong>JWT Bearer</strong> authentication.
            </p>
          </div>
        </div>

        {errorMessage && (
          <div
            id="auth-error-banner"
            role="alert"
            className="mt-4 p-3 rounded-lg badge-negative border border-hairline flex items-start gap-2.5 text-xs leading-relaxed"
          >
            <AlertCircle
              className="w-4 h-4 shrink-0 mt-0.5"
              aria-hidden="true"
            />
            <span>{errorMessage}</span>
          </div>
        )}

        <form onSubmit={handlePrimarySsoSubmit} className="mt-5 space-y-4">
          {(hasExternalOidc || devBrokerEnabled) && (
            <>
              <button
                id="btn-sso-workspace-login"
                type="submit"
                disabled={isAuthenticating}
                className="w-full py-2.5 px-4 rounded-lg btn-primary-ink text-sm font-medium inline-flex items-center justify-center gap-2 cursor-pointer disabled:opacity-60"
              >
                <UserCheck className="w-4 h-4" aria-hidden="true" />
                <span>
                  {isAuthenticating
                    ? 'Verifying Workspace · Okta SSO Assertion…'
                    : 'Continue with Google Workspace (Okta SSO)'}
                </span>
                <ArrowRight className="w-4 h-4" aria-hidden="true" />
              </button>

              {devBrokerEnabled && !hasExternalOidc && (
                <div className="rounded-lg surface-subtle border border-hairline">
                  <button
                    id="btn-toggle-sso-sandbox"
                    type="button"
                    onClick={() => setShowSandboxClaims((prev) => !prev)}
                    aria-expanded={showSandboxClaims}
                    className="w-full px-3.5 py-2 flex items-center justify-between text-xs text-ink-secondary hover:text-ink-primary cursor-pointer"
                  >
                    <span className="inline-flex items-center gap-1.5 font-medium">
                      <ShieldCheck className="w-3.5 h-3.5" aria-hidden="true" />
                      Workspace · Okta Principal Claims (Local Dev Sandbox)
                    </span>
                    {showSandboxClaims ? (
                      <ChevronUp className="w-3.5 h-3.5" aria-hidden="true" />
                    ) : (
                      <ChevronDown className="w-3.5 h-3.5" aria-hidden="true" />
                    )}
                  </button>

                  {showSandboxClaims && (
                    <div className="px-3.5 pb-3.5 pt-2 border-t border-hairline space-y-2.5">
                      <div>
                        <label
                          htmlFor="input-sso-workspace-email"
                          className="block text-[11px] font-medium text-ink-secondary mb-1"
                        >
                          Google Workspace Email (Okta Federated)
                        </label>
                        <input
                          id="input-sso-workspace-email"
                          type="email"
                          required
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          className="w-full px-2.5 py-1.5 rounded-md surface-panel border border-strong text-xs text-ink-primary tabular-nums-mono"
                        />
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                        <div>
                          <label
                            htmlFor="input-sso-display-name"
                            className="block text-[11px] font-medium text-ink-secondary mb-1"
                          >
                            Analyst Name
                          </label>
                          <input
                            id="input-sso-display-name"
                            type="text"
                            required
                            value={displayName}
                            onChange={(e) => setDisplayName(e.target.value)}
                            className="w-full px-2.5 py-1.5 rounded-md surface-panel border border-strong text-xs text-ink-primary"
                          />
                        </div>
                        <div>
                          <label
                            htmlFor="input-sso-workspace-group"
                            className="block text-[11px] font-medium text-ink-secondary mb-1"
                          >
                            Cloud Identity / Okta Groups
                          </label>
                          <input
                            id="input-sso-workspace-group"
                            type="text"
                            value={groupInput}
                            onChange={(e) => setGroupInput(e.target.value)}
                            className="w-full px-2.5 py-1.5 rounded-md surface-panel border border-strong text-xs text-ink-primary tabular-nums-mono"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </form>

        <div className="mt-5 pt-4 border-t border-hairline">
          <button
            id="btn-toggle-bearer-jwt"
            type="button"
            onClick={() => setShowBearerPanel((prev) => !prev)}
            aria-expanded={showBearerPanel}
            className="w-full flex items-center justify-between text-xs font-medium text-ink-secondary hover:text-ink-primary cursor-pointer"
          >
            <span className="inline-flex items-center gap-1.5">
              <KeyRound className="w-3.5 h-3.5" aria-hidden="true" />
              Authenticate with IAP / OIDC Bearer JWT Token
            </span>
            {showBearerPanel ? (
              <ChevronUp className="w-3.5 h-3.5" aria-hidden="true" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" aria-hidden="true" />
            )}
          </button>

          {(showBearerPanel || (!devBrokerEnabled && !hasExternalOidc)) && (
            <form onSubmit={handleBearerJwtSubmit} className="mt-3 space-y-2.5">
              <label
                htmlFor="input-iap-bearer-jwt"
                className="block text-[11px] text-ink-muted"
              >
                Paste a signed Google Cloud IAP, Google Workspace OIDC, or Okta
                SSO Bearer JWT (`Authorization: Bearer &lt;jwt&gt;`):
              </label>
              <textarea
                id="input-iap-bearer-jwt"
                rows={3}
                value={bearerJwtInput}
                onChange={(e) => setBearerJwtInput(e.target.value)}
                placeholder="eyJhbGciOiJFUzI1NiIsImtpZCI6..."
                className="w-full px-3 py-2 rounded-md surface-subtle border border-strong text-xs text-ink-primary tabular-nums-mono resize-none"
              />
              <button
                id="btn-submit-bearer-jwt"
                type="submit"
                disabled={isAuthenticating || !bearerJwtInput.trim()}
                className="w-full py-2 px-3.5 rounded-md surface-subtle border border-strong text-xs font-semibold text-ink-primary hover:opacity-90 cursor-pointer disabled:opacity-50"
              >
                Verify &amp; Connect IAP Bearer JWT
              </button>
            </form>
          )}
        </div>

        <div className="mt-5 pt-3.5 border-t border-hairline flex flex-wrap items-center justify-between gap-2 text-[11px] text-ink-muted tabular-nums-mono">
          <span>IdP: Google Workspace · Okta SSO</span>
          <span>Transport: IAP JWT (RS256 / ES256)</span>
        </div>
      </div>
    </section>
  )
}
