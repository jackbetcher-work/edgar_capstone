# Frontend — SEC EDGAR Analyst Terminal UI

React 19 + TypeScript + Tailwind CSS v4 single-page app (Vite), styled as a split-pane financial terminal. In production FastAPI serves the compiled bundle from `/`. In development, Vite proxies `/api` and `/health` to the backend.

## Run

From the repo root, `make dev` starts FastAPI (`:8000`) and this UI (`:5173`) together, and `make start` builds the bundle and serves everything from FastAPI. Inside `frontend/`:

| Script | What it does |
| :-- | :-- |
| `npm run dev` | Vite dev server on `:5173`, proxying `/api` and `/health` to `VITE_BACKEND_URL` (default `http://127.0.0.1:8000`) |
| `npm run build` | `tsc -b` type check, then the production bundle in `dist/` (FastAPI serves it) |
| `npm run preview` | Serve `dist/` on `:4173` with the same proxy |
| `npm run lint` | ESLint (also a CI gate in `cloudbuild.yaml` and GitHub Actions) |
| `npm test` | Vitest + Testing Library in jsdom (7 tests) |

## Layout

```text
src/
├── App.tsx                      # Workspace state: auth/session bootstrap, scope, chat turns, feedback, source pane
├── components/
│   ├── AuthGatePanel.tsx        # Sign-in gate (IAP session, Workspace/Okta OIDC, local dev token)
│   ├── HeaderBar.tsx            # Signed-in identity, role badge, sign-out
│   ├── HybridScopeBar.tsx       # Ticker / GICS sector / 10-K section / fiscal-year filters
│   ├── StarterPrompts.tsx       # Example longitudinal, peer-comparison, and thematic queries
│   ├── QueryComposer.tsx        # EDGR> command line ("/" to focus, <GO> to run)
│   ├── ChatTurnItem.tsx         # One answer: rendered report, variance tables, citation chips
│   ├── GroundedMarkdown.tsx     # Markdown subset (###/#### headings, bullets, bold/code/italic) + citation chips
│   ├── VarianceTableCard.tsx    # Deterministic variance table per company
│   └── SourceInspectorPane.tsx  # Right pane: cited 10-K chunk + full cached Item 1A / Item 7 section
├── services/
│   ├── authService.ts           # Bearer token storage, /api/v1/auth/* calls, OIDC fragment handling
│   └── edgarApi.ts              # /api/v1/chat, /companies, /section, /feedback (+ offline fallback data)
├── types/edgar.ts               # API response types
└── utils/formatters.ts          # USD / percent / bps formatting
```

## How answers are rendered

`/api/v1/chat` returns `answer_markdown` with inline citation tags in the form `[TICKER FYxxxx | Section Title | chunk_id](sec_filing_url)`. `GroundedMarkdown` turns each tag into a chip, and clicking a chip opens the cited chunk and its section in the source pane.

For live answers, `answer_markdown` is rendered server-side from the analyst's verified structured report:

- a headline;
- `#### Key Metrics (deterministic engine)`;
- `#### Findings`, each with an `*Evidence:*` quote;
- `#### Notes (not source-verified)`.

The response also carries `structured_report` and `report_verification` JSON. The UI doesn't use them yet; they're there for a richer findings view later. The renderer supports only the Markdown subset listed above: no tables, blockquotes, or nested lists.

If the backend is unreachable, `edgarApi.ts` falls back to bundled sample companies, variances, and citations so the UI still demos offline.

## Authentication

`App.tsx` reads `GET /api/v1/auth/config` and picks a sign-in path:

1. **IAP session (Cloud Run with IAP):** when `iap.session_auth_enabled` is set, the app probes `/api/v1/auth/me` without a token. IAP's signed `X-Goog-IAP-JWT-Assertion` header signs the user in silently. Sign Out also clears the IAP session cookie.
2. **Workspace / Okta OIDC:** redirects to the IdP, reads the ID token from the URL fragment, and exchanges it at `/api/v1/auth/sso/exchange`. The bearer JWT is stored and attached to every API call.
3. **Local development:** outside Cloud Run (`IAP_STRICT_MODE=false`), `POST /api/v1/auth/sso/dev-token` mints a local ES256 developer JWT.

A 403 from the backend's domain or RBAC policy is shown in the gate panel.

## Tests

- `src/__tests__/App.test.tsx` (3 tests): the workspace shell, a peer-comparison query with variance math, citations and the 10-K pane, and the `/` shortcut with `<GO>`.
- `src/__tests__/Auth.test.tsx` (4 tests): the SSO gate with bearer JWTs and sign-out, dev vs. strict OIDC configuration, direct IAP/OIDC bearer verification with a 403 domain error, and silent IAP-session sign-in.
