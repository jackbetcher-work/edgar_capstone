import type {
  CachedSectionResponse,
  ChatProgressEvent,
  ChatTurnResponse,
  CompaniesCatalogResponse,
  SectionFilterId,
} from '../types/edgar'
import { buildAuthorizedHeaders, clearStoredBearerToken } from './authService'
import {
  ApiAuthError,
  BackendUnavailableError,
  fetchWithAuthErrorHandling,
  requestJson,
} from './httpClient'

export { BackendUnavailableError }

let clientSessionSeq = 0

function createClientSessionId(): string {
  clientSessionSeq += 1
  const nonce = Math.random().toString(36).slice(2, 8)
  return `edgar-session-${Date.now().toString(36)}-${clientSessionSeq}-${nonce}`
}

let activeClientSessionId = createClientSessionId()

export function getClientSessionId(): string {
  return activeClientSessionId
}

/**
 * Rotate to a fresh client session ID so subsequent turns start with a clean
 * backend conversation memory, returning the previous session ID for cleanup.
 */
export function resetClientSession(): string {
  const previousSessionId = activeClientSessionId
  activeClientSessionId = createClientSessionId()
  return previousSessionId
}

export async function deleteChatSession(sessionId: string): Promise<boolean> {
  const trimmed = sessionId.trim()
  if (!trimmed) return false
  try {
    const res = await fetchWithAuthErrorHandling(
      `/api/v1/sessions/${encodeURIComponent(trimmed)}`,
      {
        method: 'DELETE',
        headers: buildAuthorizedHeaders(),
      },
      {
        onUnauthorized: clearStoredBearerToken,
        wrapNetworkError: false,
        throwOnHttpError: false,
      }
    )
    return res.ok
  } catch (err) {
    if (err instanceof ApiAuthError) {
      throw err
    }
    return false
  }
}

export interface ChatQueryPayload {
  query: string
  tickers: string
  gics_sector: string
  section_id: SectionFilterId
  start_year: number
  end_year: number
  top_k: number
  session_id: string
}

/**
 * Neutral scope sent with every chat turn. The UI has no scope filters: the
 * backend resolves companies, sector, 10-K section and fiscal years from the
 * analyst's natural-language question (plus conversation memory). These values
 * match the API defaults, which the resolver treats as "nothing pinned" — in
 * particular the 2024–2026 window is the sentinel that lets a single year
 * mentioned in the question ("in 2022") reshape the window.
 */
export const NATURAL_LANGUAGE_SCOPE = {
  tickers: 'ALL',
  gics_sector: 'ALL',
  section_id: 'ALL',
  start_year: 2024,
  end_year: 2026,
  top_k: 8,
} as const satisfies Omit<ChatQueryPayload, 'query' | 'session_id'>

export function buildChatQueryPayload(query: string): ChatQueryPayload {
  return {
    query,
    ...NATURAL_LANGUAGE_SCOPE,
    session_id: activeClientSessionId,
  }
}

export async function fetchCompaniesCatalog(): Promise<CompaniesCatalogResponse> {
  return requestJson<CompaniesCatalogResponse>(
    '/api/v1/companies',
    { headers: buildAuthorizedHeaders() },
    { onUnauthorized: clearStoredBearerToken }
  )
}

export async function executeConversationalRagQueryStream(
  query: string,
  onProgress?: (event: ChatProgressEvent) => void
): Promise<ChatTurnResponse> {
  const endpoint = '/api/v1/chat/stream'
  const res = await fetchWithAuthErrorHandling(
    endpoint,
    {
      method: 'POST',
      headers: buildAuthorizedHeaders({
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      }),
      body: JSON.stringify(buildChatQueryPayload(query)),
    },
    { onUnauthorized: clearStoredBearerToken }
  )

  if (!res.body) {
    throw new BackendUnavailableError(
      endpoint,
      res.status,
      'streaming response body was empty'
    )
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
        elapsed_ms:
          typeof parsed.elapsed_ms === 'number' ? parsed.elapsed_ms : undefined,
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
  return requestJson<CachedSectionResponse>(
    `/api/v1/section?${params.toString()}`,
    { headers: buildAuthorizedHeaders() },
    { onUnauthorized: clearStoredBearerToken }
  )
}

export async function postAnalystFeedback(
  messageId: string,
  rating: 'up' | 'down',
  comment: string,
  query: string
): Promise<boolean> {
  try {
    const res = await fetchWithAuthErrorHandling(
      '/api/v1/feedback',
      {
        method: 'POST',
        headers: buildAuthorizedHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          message_id: messageId,
          session_id: activeClientSessionId,
          rating,
          comment,
          query,
        }),
      },
      {
        onUnauthorized: clearStoredBearerToken,
        wrapNetworkError: false,
        throwOnHttpError: false,
      }
    )
    return res.ok
  } catch (err) {
    if (err instanceof ApiAuthError) {
      throw err
    }
    return false
  }
}
