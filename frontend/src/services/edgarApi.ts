import type {
  CachedSectionResponse,
  ChatTurnResponse,
  CompaniesCatalogResponse,
  SearchScopeFilters,
} from '../types/edgar'
import {
  ApiAuthError,
  buildAuthorizedHeaders,
  parseAuthErrorResponse,
} from './authService'

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

async function readErrorDetail(res: Response): Promise<string> {
  try {
    const text = await res.text()
    if (!text.trim()) return res.statusText
    try {
      const payload = JSON.parse(text) as {
        detail?: string | { message?: string }
        message?: string
      }
      if (typeof payload.detail === 'string' && payload.detail.trim()) {
        return payload.detail
      }
      if (
        payload.detail &&
        typeof payload.detail === 'object' &&
        typeof payload.detail.message === 'string'
      ) {
        return payload.detail.message
      }
      if (typeof payload.message === 'string' && payload.message.trim()) {
        return payload.message
      }
    } catch {
      // Non-JSON error body; fall through to raw text.
    }
    return text.trim().slice(0, 200)
  } catch {
    return res.statusText
  }
}

async function requestBackendJson<T>(url: string, init: RequestInit): Promise<T> {
  const endpoint = url.split('?')[0]
  let res: Response
  try {
    res = await fetch(url, init)
  } catch (err) {
    const detail = err instanceof Error && err.message ? err.message : 'network error'
    throw new BackendUnavailableError(endpoint, null, detail)
  }
  if (res.status === 401 || res.status === 403) {
    throw await parseAuthErrorResponse(res)
  }
  if (!res.ok) {
    throw new BackendUnavailableError(endpoint, res.status, await readErrorDetail(res))
  }
  try {
    return (await res.json()) as T
  } catch {
    throw new BackendUnavailableError(endpoint, res.status, 'response was not valid JSON')
  }
}

export async function fetchCompaniesCatalog(): Promise<CompaniesCatalogResponse> {
  return requestBackendJson<CompaniesCatalogResponse>('/api/v1/companies', {
    headers: buildAuthorizedHeaders(),
  })
}

const CLIENT_SESSION_ID = `edgar-session-${Date.now().toString(36)}`

export async function executeConversationalRagQuery(
  query: string,
  filters: SearchScopeFilters
): Promise<ChatTurnResponse> {
  const tickersPayload =
    filters.selectedTickers.length > 0 ? filters.selectedTickers.join(',') : 'ALL'

  return requestBackendJson<ChatTurnResponse>('/api/v1/chat', {
    method: 'POST',
    headers: buildAuthorizedHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      query,
      tickers: tickersPayload,
      gics_sector: filters.gicsSector,
      section_id: filters.sectionId,
      start_year: filters.startYear,
      end_year: filters.endYear,
      top_k: filters.topK,
      session_id: CLIENT_SESSION_ID,
    }),
  })
}

export async function fetchCached10kSection(
  ticker: string,
  fiscalYear: number,
  sectionId: 'item_1a' | 'item_7'
): Promise<CachedSectionResponse> {
  const params = new URLSearchParams({
    ticker,
    fiscal_year: String(fiscalYear),
    section_id: sectionId,
    max_chars: '38000',
  })
  return requestBackendJson<CachedSectionResponse>(
    `/api/v1/section?${params.toString()}`,
    { headers: buildAuthorizedHeaders() }
  )
}

export async function postAnalystFeedback(
  messageId: string,
  rating: 'up' | 'down',
  comment: string,
  query: string
): Promise<boolean> {
  try {
    const res = await fetch('/api/v1/feedback', {
      method: 'POST',
      headers: buildAuthorizedHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        message_id: messageId,
        session_id: CLIENT_SESSION_ID,
        rating,
        comment,
        query,
      }),
    })
    if (res.status === 401 || res.status === 403) {
      throw await parseAuthErrorResponse(res)
    }
    return res.ok
  } catch (err) {
    if (err instanceof ApiAuthError) {
      throw err
    }
    return false
  }
}
