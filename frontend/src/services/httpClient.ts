/**
 * Shared HTTP client utilities for authenticated EDGAR API requests.
 *
 * Centralizes FastAPI/GCP JSON error detail extraction, 401/403 auth error
 * translation, network-failure wrapping, and typed JSON parsing across both
 * `authService.ts` and `edgarApi.ts`.
 */

export class ApiAuthError extends Error {
  readonly status: 401 | 403

  constructor(status: 401 | 403, message: string) {
    super(message)
    this.name = 'ApiAuthError'
    this.status = status
  }
}

/**
 * Raised when the EDGAR backend cannot be reached or returns a non-success
 * response. The UI renders this message verbatim; no substitute data is ever
 * synthesized client-side.
 */
export class BackendUnavailableError extends Error {
  readonly endpoint: string
  readonly status: number | null

  constructor(endpoint: string, status: number | null, detail: string) {
    super(
      status === null
        ? `Backend unreachable — could not reach the EDGAR API (${detail}). No data was shown.`
        : `Backend error — the EDGAR API responded with HTTP ${status}${
            detail ? ` (${detail})` : ''
          }. No data was shown.`
    )
    this.name = 'BackendUnavailableError'
    this.endpoint = endpoint
    this.status = status
  }
}

export interface ExtractErrorDetailOptions {
  /** When true (default), falls back to trimmed raw text (200 chars) or statusText for non-JSON bodies. */
  fallbackToRawText?: boolean
}

/**
 * Extract a concise human-readable error message from a FastAPI/HTTP Response body.
 * Supports `{ detail: string }`, `{ detail: { message: string } }`, and `{ message: string }`.
 */
export async function extractErrorDetail(
  res: Response,
  options: ExtractErrorDetailOptions = {}
): Promise<string> {
  const { fallbackToRawText = true } = options
  try {
    const text = await res.text()
    const trimmed = text.trim()
    if (!trimmed) {
      return fallbackToRawText ? res.statusText : ''
    }
    try {
      const payload = JSON.parse(trimmed) as {
        detail?: string | { message?: string }
        message?: string
      }
      if (typeof payload.detail === 'string' && payload.detail.trim()) {
        return payload.detail.trim()
      }
      if (
        payload.detail &&
        typeof payload.detail === 'object' &&
        typeof payload.detail.message === 'string' &&
        payload.detail.message.trim()
      ) {
        return payload.detail.message.trim()
      }
      if (typeof payload.message === 'string' && payload.message.trim()) {
        return payload.message.trim()
      }
    } catch {
      // Non-JSON error body; fall through below.
    }
    return fallbackToRawText ? trimmed.slice(0, 200) : ''
  } catch {
    return fallbackToRawText ? res.statusText : ''
  }
}

export async function buildAuthError(
  res: Response,
  onUnauthorized?: () => void
): Promise<ApiAuthError> {
  const statusCode: 401 | 403 = res.status === 403 ? 403 : 401
  const fallbackMessage =
    statusCode === 403
      ? 'Access denied by Google Workspace / Cloud Identity RBAC policy.'
      : 'Authentication required. Sign in with Google Workspace (Okta SSO) or provide a valid IAP Bearer JWT.'

  const extracted = await extractErrorDetail(res, { fallbackToRawText: false })
  if (statusCode === 401) {
    onUnauthorized?.()
  }
  return new ApiAuthError(statusCode, extracted || fallbackMessage)
}

export interface HttpRequestOptions {
  /** Invoked on HTTP 401 responses before throwing `ApiAuthError` (e.g. clearing stored JWT). */
  onUnauthorized?: () => void
  /** Wrap network `fetch` failures in `BackendUnavailableError` (default: true). */
  wrapNetworkError?: boolean
  /** Throw on non-OK non-auth HTTP statuses (default: true). */
  throwOnHttpError?: boolean
  /** Optional custom error factory for non-OK non-auth HTTP responses. */
  formatHttpError?: (status: number, detail: string, endpoint: string) => Error
}

export async function fetchWithAuthErrorHandling(
  url: string,
  init?: RequestInit,
  options: HttpRequestOptions = {}
): Promise<Response> {
  const {
    onUnauthorized,
    wrapNetworkError = true,
    throwOnHttpError = true,
    formatHttpError,
  } = options
  const endpoint = url.split('?')[0]

  let res: Response
  try {
    res = init !== undefined ? await fetch(url, init) : await fetch(url)
  } catch (err) {
    if (!wrapNetworkError) {
      throw err
    }
    const detail = err instanceof Error && err.message ? err.message : 'network error'
    throw new BackendUnavailableError(endpoint, null, detail)
  }

  if (res.status === 401 || res.status === 403) {
    throw await buildAuthError(res, onUnauthorized)
  }

  if (throwOnHttpError && !res.ok) {
    const detail = await extractErrorDetail(res)
    if (formatHttpError) {
      throw formatHttpError(res.status, detail, endpoint)
    }
    throw new BackendUnavailableError(endpoint, res.status, detail)
  }

  return res
}

export async function requestJson<T>(
  url: string,
  init?: RequestInit,
  options: HttpRequestOptions = {}
): Promise<T> {
  const endpoint = url.split('?')[0]
  const res = await fetchWithAuthErrorHandling(url, init, options)
  try {
    return (await res.json()) as T
  } catch {
    throw new BackendUnavailableError(
      endpoint,
      res.status,
      'response was not valid JSON'
    )
  }
}
