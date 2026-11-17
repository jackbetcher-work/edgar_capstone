import type {
  CachedSectionResponse,
  ChatProgressEvent,
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

export async function executeConversationalRagQueryStream(
  query: string,
  filters: SearchScopeFilters,
  onProgress?: (event: ChatProgressEvent) => void
): Promise<ChatTurnResponse> {
  const endpoint = '/api/v1/chat/stream'
  const tickersPayload =
    filters.selectedTickers.length > 0 ? filters.selectedTickers.join(',') : 'ALL'

  let res: Response
  try {
    res = await fetch(endpoint, {
      method: 'POST',
      headers: buildAuthorizedHeaders({
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      }),
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

  const contentType = (res.headers.get('content-type') || '').toLowerCase()
  if (contentType.includes('application/json') || !res.body) {
    try {
      return (await res.json()) as ChatTurnResponse
    } catch {
      throw new BackendUnavailableError(endpoint, res.status, 'response was not valid JSON')
    }
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finalResult: ChatTurnResponse | null = null

  const processBlock = (block: string) => {
    const trimmed = block.trim()
    if (!trimmed || trimmed.startsWith(':')) return
    let eventName = 'message'
    const dataLines: string[] = []
    for (const line of trimmed.split(/\r?\n/)) {
      if (line.startsWith(':')) continue
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart())
      }
    }
    if (dataLines.length === 0) return
    const rawData = dataLines.join('\n')
    let parsed: Record<string, unknown>
    try {
      parsed = JSON.parse(rawData) as Record<string, unknown>
    } catch {
      return
    }
    if (eventName === 'progress') {
      onProgress?.({
        stage: String(parsed.stage ?? 'working'),
        message: String(parsed.message ?? ''),
        meta: (parsed.meta as Record<string, unknown> | null) ?? null,
        elapsed_ms: typeof parsed.elapsed_ms === 'number' ? parsed.elapsed_ms : undefined,
      })
    } else if (eventName === 'result') {
      finalResult = parsed as unknown as ChatTurnResponse
    } else if (eventName === 'error') {
      const msg =
        typeof parsed.message === 'string' && parsed.message.trim()
          ? parsed.message
          : String(parsed.error ?? 'Stream error')
      throw new BackendUnavailableError(endpoint, 503, msg)
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (value) {
      buffer += decoder.decode(value, { stream: !done })
      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() ?? ''
      for (const blk of blocks) {
        processBlock(blk)
      }
    }
    if (done) {
      if (buffer.trim()) {
        processBlock(buffer)
      }
      break
    }
  }

  if (!finalResult) {
    throw new BackendUnavailableError(
      endpoint,
      res.status,
      'stream ended before emitting a final grounded result'
    )
  }
  return finalResult
}

export interface UserSessionSummary {
  session_id: string
  user_id: string
  turn_count: number
  remembered_tickers: string[]
  last_gics_sector: string
  last_start_year: number
  last_end_year: number
  updated_at: string
}

export async function listUserSessions(): Promise<{
  status: string
  sessions: UserSessionSummary[]
}> {
  return requestBackendJson('/api/v1/sessions', {
    headers: buildAuthorizedHeaders(),
  })
}

export async function deleteUserSession(sessionId: string): Promise<void> {
  await requestBackendJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    headers: buildAuthorizedHeaders(),
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
