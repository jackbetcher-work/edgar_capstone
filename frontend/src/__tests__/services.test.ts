import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  buildAuthorizedHeaders,
  buildExternalOidcAuthorizeUrl,
  clearStoredBearerToken,
  consumeOidcFragmentIdToken,
  exchangeBearerTokenSession,
  fetchAuthDiscoveryConfig,
  getStoredBearerToken,
  logoutAuthenticatedSession,
  setStoredBearerToken,
} from '../services/authService'
import {
  BackendUnavailableError,
  buildChatQueryPayload,
  executeConversationalRagQueryStream,
  fetchCached10kSection,
  postAnalystFeedback,
} from '../services/edgarApi'
import {
  ApiAuthError,
  buildAuthError,
  extractErrorDetail,
  fetchWithAuthErrorHandling,
  requestJson,
} from '../services/httpClient'
import type { AuthDiscoveryConfig, ChatProgressEvent, SearchScopeFilters } from '../types/edgar'

const DEFAULT_FILTERS: SearchScopeFilters = {
  selectedTickers: ['NVDA', 'AAPL'],
  gicsSector: 'Information Technology',
  sectionId: 'item_7',
  startYear: 2024,
  endYear: 2026,
  topK: 8,
}

function makeChunkedSseResponse(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
  return new Response(stream, {
    status,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

describe('httpClient utilities', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('extractErrorDetail parses detail string, nested detail.message, message, raw text, and empty body', async () => {
    expect(
      await extractErrorDetail(new Response(JSON.stringify({ detail: ' Bad scope ' }), { status: 400 }))
    ).toBe('Bad scope')
    expect(
      await extractErrorDetail(
        new Response(JSON.stringify({ detail: { message: 'Nested detail' } }), { status: 400 })
      )
    ).toBe('Nested detail')
    expect(
      await extractErrorDetail(new Response(JSON.stringify({ message: 'Top message' }), { status: 500 }))
    ).toBe('Top message')
    expect(
      await extractErrorDetail(new Response('plain text failure', { status: 502 }))
    ).toBe('plain text failure')
    expect(
      await extractErrorDetail(new Response('plain text failure', { status: 502 }), {
        fallbackToRawText: false,
      })
    ).toBe('')
    expect(
      await extractErrorDetail(new Response('', { status: 503, statusText: 'Service Unavailable' }))
    ).toBe('Service Unavailable')
  })

  it('buildAuthError invokes onUnauthorized on 401 but NOT on 403', async () => {
    const onUnauthorized = vi.fn()
    const err401 = await buildAuthError(
      new Response(JSON.stringify({ detail: 'Token expired' }), { status: 401 }),
      onUnauthorized
    )
    expect(err401).toBeInstanceOf(ApiAuthError)
    expect(err401.status).toBe(401)
    expect(err401.message).toBe('Token expired')
    expect(onUnauthorized).toHaveBeenCalledTimes(1)

    onUnauthorized.mockClear()
    const err403 = await buildAuthError(new Response('<html>forbidden</html>', { status: 403 }), onUnauthorized)
    expect(err403.status).toBe(403)
    expect(err403.message).toContain('Access denied')
    expect(onUnauthorized).not.toHaveBeenCalled()
  })

  it('wraps network errors in BackendUnavailableError unless wrapNetworkError=false', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(fetchWithAuthErrorHandling('/api/v1/companies')).rejects.toMatchObject({
      name: 'BackendUnavailableError',
      endpoint: '/api/v1/companies',
      status: null,
    })
    await expect(
      fetchWithAuthErrorHandling('/api/v1/companies', undefined, { wrapNetworkError: false })
    ).rejects.toThrow('Failed to fetch')
  })

  it('requestJson throws BackendUnavailableError on non-JSON 200 payload and custom formatHttpError', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response('not-json', { status: 200, headers: { 'Content-Type': 'application/json' } })
    )
    await expect(requestJson('/api/v1/companies')).rejects.toThrow('response was not valid JSON')

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'boom' }), { status: 500 })
    )
    await expect(
      fetchWithAuthErrorHandling('/api/v1/companies', undefined, {
        formatHttpError: (status, detail, ep) => new Error(`${ep}:${status}:${detail}`),
      })
    ).rejects.toThrow('/api/v1/companies:500:boom')
  })
})

describe('edgarApi SSE parsing & payloads', () => {
  beforeEach(() => {
    clearStoredBearerToken()
    vi.restoreAllMocks()
  })

  it('buildChatQueryPayload formats selected tickers or ALL fallback', () => {
    const withTickers = buildChatQueryPayload('Compare NVDA and AAPL', DEFAULT_FILTERS)
    expect(withTickers.tickers).toBe('NVDA,AAPL')
    expect(withTickers.gics_sector).toBe('Information Technology')
    expect(withTickers.section_id).toBe('item_7')
    expect(withTickers.start_year).toBe(2024)
    expect(withTickers.end_year).toBe(2026)
    expect(withTickers.session_id).toMatch(/^edgar-session-/)

    const allTickers = buildChatQueryPayload('All sector risks', {
      ...DEFAULT_FILTERS,
      selectedTickers: [],
    })
    expect(allTickers.tickers).toBe('ALL')
  })

  it('parses chunked SSE streams across arbitrary byte splits, comments, multi-line data, and ignores malformed JSON', async () => {
    const progressEvents: ChatProgressEvent[] = []
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      makeChunkedSseResponse([
        ': keep-alive\n\n',
        'event: progress\ndata: {"stage":"connected",',
        '"message":"Connected","elapsed_ms":12}\n\n',
        'event: progress\ndata: {broken json\n\n',
        'event: result\n: comment inside frame\n',
        'data: {"status":"success",\n',
        'data: "intent":"longitudinal_single_company",\n',
        'data: "answer_markdown":"ok"}\n\n',
      ])
    )

    const res = await executeConversationalRagQueryStream(
      'Show NVDA revenue',
      DEFAULT_FILTERS,
      (ev) => progressEvents.push(ev)
    )
    expect(progressEvents).toHaveLength(1)
    expect(progressEvents[0]).toMatchObject({ stage: 'connected', elapsed_ms: 12 })
    expect(res.status).toBe('success')
    expect(res.answer_markdown).toBe('ok')
  })

  it('throws BackendUnavailableError(503) on SSE error event and on premature EOF without result', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      makeChunkedSseResponse([
        'event: error\ndata: {"error":"internal_server_error","message":"Stream backend failed"}\n\n',
      ])
    )
    await expect(
      executeConversationalRagQueryStream('q', DEFAULT_FILTERS)
    ).rejects.toMatchObject({
      name: 'BackendUnavailableError',
      status: 503,
      message: expect.stringContaining('Stream backend failed'),
    })

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      makeChunkedSseResponse(['event: progress\ndata: {"stage":"working","message":"..."}\n\n'])
    )
    await expect(
      executeConversationalRagQueryStream('q', DEFAULT_FILTERS)
    ).rejects.toThrow('stream ended before emitting a final grounded result')
  })

  it('supports non-SSE JSON fallback and rejects invalid JSON fallback', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(JSON.stringify({ status: 'success', answer_markdown: 'json fallback' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    const ok = await executeConversationalRagQueryStream('q', DEFAULT_FILTERS)
    expect(ok.answer_markdown).toBe('json fallback')

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response('not-json', { status: 200, headers: { 'Content-Type': 'application/json' } })
    )
    await expect(executeConversationalRagQueryStream('q', DEFAULT_FILTERS)).rejects.toBeInstanceOf(
      BackendUnavailableError
    )
  })

  it('fetchCached10kSection sends expected query params and postAnalystFeedback handles ok, network error, and 401', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(JSON.stringify({ status: 'success', ticker: 'NVDA', fiscal_year: 2026 }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    await fetchCached10kSection('NVDA', 2026, 'item_1a')
    expect(fetchSpy.mock.calls[0][0]).toBe(
      '/api/v1/section?ticker=NVDA&fiscal_year=2026&section_id=item_1a&max_chars=38000'
    )

    // postAnalystFeedback: 200 -> true, 500 -> false, network error -> false, 401 -> rethrows ApiAuthError
    fetchSpy.mockResolvedValueOnce(new Response('{}', { status: 200 }))
    expect(await postAnalystFeedback('m1', 'up', 'great', 'q')).toBe(true)

    fetchSpy.mockResolvedValueOnce(new Response('err', { status: 500 }))
    expect(await postAnalystFeedback('m1', 'down', 'bad', 'q')).toBe(false)

    fetchSpy.mockRejectedValueOnce(new TypeError('offline'))
    expect(await postAnalystFeedback('m1', 'up', '', 'q')).toBe(false)

    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'expired' }), { status: 401 })
    )
    await expect(postAnalystFeedback('m1', 'up', '', 'q')).rejects.toBeInstanceOf(ApiAuthError)
  })
})

describe('authService helpers', () => {
  beforeEach(() => {
    clearStoredBearerToken()
    window.history.replaceState(null, '', '/')
    vi.restoreAllMocks()
  })

  it('consumes id_token from URL hash fragment and clears fragment without reload', () => {
    expect(consumeOidcFragmentIdToken()).toBeNull()
    window.location.hash = '#id_token=eyJ.abc.sig&state=xyz'
    expect(consumeOidcFragmentIdToken()).toBe('eyJ.abc.sig')
    expect(window.location.hash).toBe('')
  })

  it('buildExternalOidcAuthorizeUrl constructs OIDC implicit URL with hd and login_hint', () => {
    const baseCfg: AuthDiscoveryConfig = {
      status: 'ok',
      auth_required: true,
      access_pattern: 'iap_direct',
      identity_provider: {
        name: 'Google Workspace + Okta SSO',
        protocol: 'OIDC',
        google_issuer: 'https://accounts.google.com',
        iap_issuer: 'https://cloud.google.com/iap',
        okta_issuer: null,
        authorization_endpoint: 'https://accounts.google.com/o/oauth2/v2/auth',
        google_client_id: 'client-123',
        okta_client_id: null,
        allowed_workspace_domains: ['example.com'],
        required_analyst_groups: [],
        local_sso_dev_broker_enabled: false,
      },
      iap: {
        enabled: true,
        strict_production_mode: true,
        bearer_header: 'Authorization',
        assertion_header: 'X-Goog-IAP-JWT-Assertion',
        audience_configured: true,
      },
    }
    const url = buildExternalOidcAuthorizeUrl(baseCfg, ' analyst@example.com ')
    expect(url).toBeTruthy()
    const parsed = new URL(url!)
    expect(parsed.searchParams.get('client_id')).toBe('client-123')
    expect(parsed.searchParams.get('hd')).toBe('example.com')
    expect(parsed.searchParams.get('login_hint')).toBe('analyst@example.com')
    expect(parsed.searchParams.get('nonce')).toMatch(/^[0-9a-f]{32}$/)

    // Missing endpoint or client id returns null
    expect(
      buildExternalOidcAuthorizeUrl({
        ...baseCfg,
        identity_provider: { ...baseCfg.identity_provider, authorization_endpoint: null },
      })
    ).toBeNull()
  })

  it('strips leading Bearer prefix during token exchange and tolerates sessionStorage errors', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          access_token: 'issued-jwt',
          user: { email: 'a@example.com', name: 'A', roles: ['analyst'] },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    )
    const user = await exchangeBearerTokenSession('  Bearer raw.jwt.val  ')
    expect(user.email).toBe('a@example.com')
    const body = JSON.parse(String(fetchSpy.mock.calls[0][1]?.body))
    expect(body).toEqual({ id_token: 'raw.jwt.val' })
    expect(getStoredBearerToken()).toBe('issued-jwt')
    expect(buildAuthorizedHeaders({ 'X-Test': '1' })).toEqual({
      'X-Test': '1',
      Authorization: 'Bearer issued-jwt',
    })

    // Logout clears stored token even when backend fetch fails
    fetchSpy.mockRejectedValueOnce(new Error('network down'))
    await logoutAuthenticatedSession()
    expect(getStoredBearerToken()).toBeNull()

    // sessionStorage failure never crashes token setter/getter
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('storage blocked')
    })
    setStoredBearerToken('mem-only-token')
    expect(getStoredBearerToken()).toBe('mem-only-token')
  })

  it('fetchAuthDiscoveryConfig handles non-OK and network failures gracefully', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(new Response('err', { status: 500 }))
    expect(await fetchAuthDiscoveryConfig()).toEqual({ backendReachable: false, config: null })
    vi.spyOn(globalThis, 'fetch').mockRejectedValueOnce(new Error('down'))
    expect(await fetchAuthDiscoveryConfig()).toEqual({ backendReachable: false, config: null })
  })
})
