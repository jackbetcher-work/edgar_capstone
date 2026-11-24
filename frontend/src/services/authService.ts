import type {
  AuthDiscoveryConfig,
  AuthenticatedUserProfile,
  SsoSessionResponse,
} from '../types/edgar'
import { ApiAuthError, buildAuthError, requestJson } from './httpClient'

export { ApiAuthError }

const SESSION_STORAGE_JWT_KEY = 'edgar_iap_bearer_jwt'

let inMemoryBearerJwt: string | null = null

export function getStoredBearerToken(): string | null {
  if (inMemoryBearerJwt) {
    return inMemoryBearerJwt
  }
  try {
    const stored = window.sessionStorage.getItem(SESSION_STORAGE_JWT_KEY)
    if (stored && stored.trim()) {
      inMemoryBearerJwt = stored.trim()
      return inMemoryBearerJwt
    }
  } catch {
    // Ignore storage access errors in restricted environments
  }
  return null
}

export function setStoredBearerToken(token: string | null): void {
  const cleaned = token?.trim() || null
  inMemoryBearerJwt = cleaned
  try {
    if (cleaned) {
      window.sessionStorage.setItem(SESSION_STORAGE_JWT_KEY, cleaned)
    } else {
      window.sessionStorage.removeItem(SESSION_STORAGE_JWT_KEY)
    }
  } catch {
    // Ignore storage access errors
  }
}

export function clearStoredBearerToken(): void {
  setStoredBearerToken(null)
}

export function buildAuthorizedHeaders(
  extraHeaders: Record<string, string> = {}
): Record<string, string> {
  const headers: Record<string, string> = { ...extraHeaders }
  const token = getStoredBearerToken()
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }
  return headers
}

export async function parseAuthErrorResponse(
  res: Response
): Promise<ApiAuthError> {
  return buildAuthError(res, clearStoredBearerToken)
}

export async function fetchAuthDiscoveryConfig(): Promise<{
  backendReachable: boolean
  config: AuthDiscoveryConfig | null
}> {
  try {
    const res = await fetch('/api/v1/auth/config')
    if (!res.ok) {
      return { backendReachable: false, config: null }
    }
    const config = (await res.json()) as AuthDiscoveryConfig
    return { backendReachable: true, config }
  } catch {
    return { backendReachable: false, config: null }
  }
}

export function consumeOidcFragmentIdToken(): string | null {
  try {
    const hash = window.location.hash
    if (!hash || !hash.includes('id_token=')) {
      return null
    }
    const params = new URLSearchParams(hash.replace(/^#/, ''))
    const idToken = params.get('id_token')
    if (idToken && idToken.trim()) {
      window.history.replaceState(
        null,
        document.title,
        window.location.pathname + window.location.search
      )
      return idToken.trim()
    }
  } catch {
    // Ignore history API errors
  }
  return null
}

export function buildExternalOidcAuthorizeUrl(
  config: AuthDiscoveryConfig,
  loginHint?: string
): string | null {
  const endpoint = config.identity_provider.authorization_endpoint
  const clientId = config.identity_provider.google_client_id
  if (!endpoint || !clientId) {
    return null
  }

  const nonceBytes = new Uint8Array(16)
  window.crypto.getRandomValues(nonceBytes)
  const nonce = Array.from(nonceBytes, (b) => b.toString(16).padStart(2, '0')).join('')

  const params = new URLSearchParams({
    client_id: clientId,
    redirect_uri: window.location.origin + window.location.pathname,
    response_type: 'id_token',
    scope: 'openid email profile groups',
    nonce,
    state: nonce.slice(0, 16),
  })

  const allowedDomains = config.identity_provider.allowed_workspace_domains
  if (allowedDomains.length === 1) {
    params.set('hd', allowedDomains[0])
  }
  if (loginHint?.trim()) {
    params.set('login_hint', loginHint.trim())
  }

  return `${endpoint}?${params.toString()}`
}

export async function exchangeBearerTokenSession(
  rawJwt: string
): Promise<AuthenticatedUserProfile> {
  const cleaned = rawJwt.trim().replace(/^Bearer\s+/i, '').trim()
  const body = await requestJson<SsoSessionResponse>(
    '/api/v1/auth/sso/exchange',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id_token: cleaned }),
    },
    {
      onUnauthorized: clearStoredBearerToken,
      wrapNetworkError: false,
      formatHttpError: (status) =>
        new Error(`SSO token verification failed (HTTP ${status}).`),
    }
  )
  setStoredBearerToken(body.access_token)
  return body.user
}

export async function loginWithWorkspaceOktaDevBroker(params: {
  email?: string
  name?: string
  groups?: string[]
}): Promise<AuthenticatedUserProfile> {
  const body = await requestJson<SsoSessionResponse>(
    '/api/v1/auth/sso/dev-token',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: params.email ?? 'analyst@fde-workspace.corp',
        name: params.name ?? 'Senior Financial Analyst',
        groups: params.groups ?? ['sec-edgar-analysts'],
      }),
    },
    {
      onUnauthorized: clearStoredBearerToken,
      wrapNetworkError: false,
      formatHttpError: (status) =>
        new Error(`Workspace Okta SSO handshake failed (HTTP ${status}).`),
    }
  )
  setStoredBearerToken(body.access_token)
  return body.user
}

export async function fetchAuthenticatedUser(): Promise<AuthenticatedUserProfile> {
  const body = await requestJson<{ user: AuthenticatedUserProfile }>(
    '/api/v1/auth/me',
    {
      headers: buildAuthorizedHeaders(),
    },
    {
      onUnauthorized: clearStoredBearerToken,
      wrapNetworkError: false,
      formatHttpError: (status) =>
        new Error(`Unable to verify current session (HTTP ${status}).`),
    }
  )
  return body.user
}

export async function logoutAuthenticatedSession(): Promise<void> {
  const token = getStoredBearerToken()
  try {
    if (token) {
      await fetch('/api/v1/auth/logout', {
        method: 'POST',
        headers: buildAuthorizedHeaders(),
      })
    }
  } catch {
    // Clear local session regardless of network reachability
  } finally {
    clearStoredBearerToken()
  }
}
