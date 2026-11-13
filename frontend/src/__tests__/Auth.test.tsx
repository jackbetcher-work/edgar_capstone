import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from '../App'
import { clearStoredBearerToken } from '../services/authService'

const MOCK_DEV_AUTH_CONFIG = {
  status: 'ok',
  auth_required: true,
  access_pattern: 'iap_jwt_bearer',
  identity_provider: {
    name: 'Google Workspace (Okta SSO)',
    protocol: 'OIDC / SAML-Federated Workspace + IAP',
    google_issuer: 'https://accounts.google.com',
    iap_issuer: 'https://cloud.google.com/iap',
    okta_issuer: 'https://acme.okta.com/oauth2/default',
    okta_client_id: null,
    google_client_id: null,
    authorization_endpoint: null,
    allowed_workspace_domains: ['fde-workspace.corp'],
    required_analyst_groups: ['sec-edgar-analysts'],
    local_sso_dev_broker_enabled: true,
  },
  iap: {
    enabled: true,
    strict_production_mode: false,
    bearer_header: 'Authorization',
    assertion_header: 'X-Goog-IAP-JWT-Assertion',
    audience_configured: true,
  },
}

const MOCK_PROD_AUTH_CONFIG = {
  ...MOCK_DEV_AUTH_CONFIG,
  identity_provider: {
    ...MOCK_DEV_AUTH_CONFIG.identity_provider,
    okta_client_id: 'okta-prod-client-id',
    authorization_endpoint: 'https://acme.okta.com/oauth2/default/v1/authorize',
    local_sso_dev_broker_enabled: false,
  },
  iap: {
    ...MOCK_DEV_AUTH_CONFIG.iap,
    strict_production_mode: true,
  },
}

const MOCK_AUTHENTICATED_USER = {
  sub: 'okta_workspace:analyst@fde-workspace.corp',
  email: 'analyst@fde-workspace.corp',
  name: 'Senior Financial Analyst',
  hosted_domain: 'fde-workspace.corp',
  identity_provider: 'okta_sso',
  sso_federation: 'google_workspace_okta_sso',
  issuer: 'https://acme.okta.com/oauth2/default',
  audience: '/projects/12345/global/backendServices/67890',
  groups: ['sec-edgar-analysts'],
  roles: ['authenticated', 'analyst'],
  issued_at: 1794500000,
  expires_at: 1794503600,
  iap_assertion_verified: true,
}

const MOCK_COMPANIES_CATALOG = {
  status: 'success',
  company_count: 1,
  total_profiles: 7,
  min_year: 2020,
  max_year: 2026,
  gics_sectors: ['Information Technology'],
  companies: [
    {
      ticker: 'NVDA',
      company_name: 'NVIDIA CORP',
      cik: '0001045810',
      gics_sector: 'Information Technology',
      gics_sub_industry: 'Semiconductors',
      fiscal_years: [2024, 2025, 2026],
      latest_fiscal_year: 2026,
      latest_revenue: 215_938_000_000,
      latest_operating_income: 130_387_000_000,
      latest_net_income: 120_067_000_000,
      latest_operating_margin_pct: 60.38,
      latest_net_margin_pct: 55.6,
      sec_filing_url: 'https://www.sec.gov/Archives/edgar/data/1045810/nvda.htm',
    },
  ],
}

describe('Google Workspace (Okta SSO) & IAP JWT Bearer Authentication UI', () => {
  beforeEach(() => {
    clearStoredBearerToken()
    vi.restoreAllMocks()
  })

  afterEach(() => {
    clearStoredBearerToken()
    cleanup()
  })

  it('gates the workspace behind Google Workspace Okta/SSO login, attaches Bearer JWTs, and supports Sign Out', async () => {
    const seenAuthHeaders: Record<string, string | undefined> = {}

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = typeof input === 'string' ? input : input.toString()
      const headers = (init?.headers ?? {}) as Record<string, string>
      seenAuthHeaders[url] = headers.Authorization

      if (url === '/api/v1/auth/config') {
        return new Response(JSON.stringify(MOCK_DEV_AUTH_CONFIG), { status: 200 })
      }
      if (url === '/api/v1/auth/sso/dev-token') {
        return new Response(
          JSON.stringify({
            status: 'authenticated',
            token_type: 'Bearer',
            access_token: 'signed.iap.es256.jwt',
            expires_at: 1794503600,
            user: MOCK_AUTHENTICATED_USER,
          }),
          { status: 200 }
        )
      }
      if (url === '/api/v1/companies') {
        if (headers.Authorization !== 'Bearer signed.iap.es256.jwt') {
          return new Response(
            JSON.stringify({ detail: { message: 'Missing IAP Bearer JWT' } }),
            { status: 401 }
          )
        }
        return new Response(JSON.stringify(MOCK_COMPANIES_CATALOG), {
          status: 200,
        })
      }
      if (url === '/api/v1/auth/logout') {
        return new Response(JSON.stringify({ status: 'logged_out' }), {
          status: 200,
        })
      }
      return new Response('{}', { status: 404 })
    })

    render(<App />)

    // Verify single h1 and IAP Sign-In Gate rendered without dev-only sandbox clutter
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
    expect(
      await screen.findByText(/Sign in with Google Workspace/i)
    ).toBeTruthy()
    expect(screen.getByText(/AUTH LOCKED/i)).toBeTruthy()
    expect(screen.queryByText(/Local Dev Sandbox/i)).toBeNull()

    // Sign in via Google Workspace (Okta SSO)
    fireEvent.click(
      screen.getByRole('button', {
        name: /Continue with Google Workspace \(Okta SSO\)/i,
      })
    )

    // Workspace unlocks and HeaderBar displays verified principal email
    await waitFor(() => {
      expect(
        screen.getByText(/Query audited financials and grounded 10-K disclosures/i)
      ).toBeTruthy()
    })
    expect(screen.getByText('analyst@fde-workspace.corp')).toBeTruthy()
    expect(screen.getByText('ANALYST')).toBeTruthy()
    expect(seenAuthHeaders['/api/v1/companies']).toBe(
      'Bearer signed.iap.es256.jwt'
    )

    // Sign Out revokes session and returns to SSO gate
    fireEvent.click(screen.getByRole('button', { name: /Sign Out/i }))
    await waitFor(() => {
      expect(screen.getByText(/Sign in with Google Workspace/i)).toBeTruthy()
    })
    expect(seenAuthHeaders['/api/v1/auth/logout']).toBe(
      'Bearer signed.iap.es256.jwt'
    )
  })

  it('renders identical login controls in dev and strict production OIDC configurations', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url === '/api/v1/auth/config') {
        return new Response(JSON.stringify(MOCK_PROD_AUTH_CONFIG), { status: 200 })
      }
      return new Response('{}', { status: 404 })
    })

    render(<App />)
    expect(
      await screen.findByRole('button', {
        name: /Continue with Google Workspace \(Okta SSO\)/i,
      })
    ).toBeTruthy()
    expect(
      screen.getByRole('button', {
        name: /Authenticate with IAP \/ OIDC Bearer JWT Token/i,
      })
    ).toBeTruthy()
    expect(screen.queryByText(/Local Dev Sandbox/i)).toBeNull()
  })

  it('supports direct IAP / OIDC Bearer JWT verification and surfaces 403 domain policy errors', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url === '/api/v1/auth/config') {
        return new Response(JSON.stringify(MOCK_DEV_AUTH_CONFIG), { status: 200 })
      }
      if (url === '/api/v1/auth/sso/exchange') {
        const body = JSON.parse(String(init?.body ?? '{}')) as {
          id_token?: string
        }
        if (body.id_token === 'forbidden.domain.jwt') {
          return new Response(
            JSON.stringify({
              detail: {
                message:
                  "Google Workspace domain 'unauthorized.org' is not authorized for SEC EDGAR internal APIs.",
              },
            }),
            { status: 403 }
          )
        }
        return new Response(
          JSON.stringify({
            status: 'authenticated',
            token_type: 'Bearer',
            access_token: body.id_token,
            expires_at: 1794503600,
            user: MOCK_AUTHENTICATED_USER,
          }),
          { status: 200 }
        )
      }
      if (url === '/api/v1/companies') {
        return new Response(JSON.stringify(MOCK_COMPANIES_CATALOG), {
          status: 200,
        })
      }
      return new Response('{}', { status: 404 })
    })

    render(<App />)
    const toggleJwtBtn = await screen.findByRole('button', {
      name: /Authenticate with IAP \/ OIDC Bearer JWT Token/i,
    })
    fireEvent.click(toggleJwtBtn)

    const jwtTextarea = screen.getByLabelText(/Paste a signed Google Cloud IAP/i)
    fireEvent.change(jwtTextarea, { target: { value: 'forbidden.domain.jwt' } })
    fireEvent.click(
      screen.getByRole('button', { name: /Verify & Connect IAP Bearer JWT/i })
    )

    const alertBanner = await screen.findByRole('alert')
    expect(alertBanner.textContent).toContain('unauthorized.org')

    // Now submit a valid IAP Bearer JWT
    fireEvent.change(jwtTextarea, {
      target: { value: 'valid.programmatic.iap.jwt' },
    })
    fireEvent.click(
      screen.getByRole('button', { name: /Verify & Connect IAP Bearer JWT/i })
    )

    await waitFor(() => {
      expect(screen.getByText('analyst@fde-workspace.corp')).toBeTruthy()
    })
  })
})
