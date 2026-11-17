import type {
  CachedSectionResponse,
  ChatProgressEvent,
  ChatTurnResponse,
  CompaniesCatalogResponse,
  SearchScopeFilters,
} from '../types/edgar'
import { buildAuthorizedHeaders, clearStoredBearerToken } from './authService'
import {
  ApiAuthError,
  BackendUnavailableError,
  fetchWithAuthErrorHandling,
  requestJson,
} from './httpClient'

export { BackendUnavailableError }

const CLIENT_SESSION_ID = `edgar-session-${Date.now().toString(36)}`

export interface ChatQueryPayload {
  query: string
  tickers: string
  gics_sector: string
  section_id: SearchScopeFilters['sectionId']
  start_year: number
  end_year: number
  top_k: number
  session_id: string
}

export function buildChatQueryPayload(
  query: string,
  filters: SearchScopeFilters
): ChatQueryPayload {
  return {
    query,
    tickers:
      filters.selectedTickers.length > 0
        ? filters.selectedTickers.join(',')
        : 'ALL',
    gics_sector: filters.gicsSector,
    section_id: filters.sectionId,
    start_year: filters.startYear,
    end_year: filters.endYear,
    top_k: filters.topK,
    session_id: CLIENT_SESSION_ID,
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
  filters: SearchScopeFilters,
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
      body: JSON.stringify(buildChatQueryPayload(query, filters)),
    },
    { onUnauthorized: clearStoredBearerToken }
  )

  const contentType = (res.headers.get('content-type') || '').toLowerCase()
  if (contentType.includes('application/json') || !res.body) {
    try {
      return (await res.json()) as ChatTurnResponse
    } catch {
      throw new BackendUnavailableError(
        endpoint,
        res.status,
        'response was not valid JSON'
      )
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
          session_id: CLIENT_SESSION_ID,
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
