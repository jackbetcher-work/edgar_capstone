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
| `npm test` | Vitest + Testing Library in jsdom (14 tests across `App.test.tsx`, `Auth.test.tsx`, `GroundingFactCards.test.tsx`) |

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
│   ├── ChatTurnItem.tsx         # One answer: live SSE progress bar, coverage notices banner, sector coverage pill, suggested follow-up chips, rendered report, variance/flag/delta cards
│   ├── DisclosureDeltaCard.tsx  # Deterministic YoY wording deltas (added / removed / modified passages)
│   ├── GroundedMarkdown.tsx     # Markdown subset (###/#### headings, blockquotes, bullets, bold/code/italic) + clickable inline text & XBRL citation chips
│   ├── MaterialFlagsCard.tsx    # Deterministic anomaly flags with period_of_report, xbrl_concept & clickable [XBRL] triggers
│   ├── VarianceTableCard.tsx    # Deterministic variance table per company with period_of_report dates, DIFFERING FY-END DATES badge & [XBRL] triggers
│   └── SourceInspectorPane.tsx  # Right pane: 3 tabs — Cited Chunk, Full 10-K Section, and XBRL Fact provenance tab
├── services/
│   ├── authService.ts           # Bearer token storage, /api/v1/auth/* calls, OIDC fragment handling
│   └── edgarApi.ts              # SSE streaming (/api/v1/chat/stream), /api/v1/chat, /sessions, /companies, /section, /feedback (live backend only; no offline fallback)
├── types/edgar.ts               # API response, SSE progress, NumericCitation, CoverageNotice & XBRL types
└── utils/formatters.ts          # USD / percent / bps / period-end date formatting, combined text+XBRL citation parser & status badges
```

## How answers are rendered

The UI streams turns over **`POST /api/v1/chat/stream`** (SSE) with live progress events (`connected`, `scope_resolved`, `sector_map_progress`, `synthesizing`, `result`, `error`). `POST /api/v1/chat` remains supported as a non-streaming endpoint.

Responses return `answer_markdown` containing:
- **Inline text citation tags:** `[TICKER FYxxxx | Section Title | chunk_id](sec_filing_url)` — opens the **Cited Chunk** / **Full Section** view in `SourceInspectorPane`.
- **Inline XBRL numeric citation tags:** `[TICKER FYxxxx | XBRL <concept> | acc:<accession_number>](sec_filing_url)` — opens the **XBRL Fact** tab in `SourceInspectorPane` showing accession numbers, XBRL concepts, denominator concepts, derived calculation inputs, SEC EDGAR filing links, and an MD&A jump button.
- **Deterministic blockquotes (`coverage_notices`):** explicit notices for missing fiscal years (e.g., calendar-year filers whose FY2026 10-Ks file in Q1 2027), fiscal-calendar misalignment (`DIFFERING FY-END DATES`), and full-sector map-reduce coverage counts.

For verified live answers, `answer_markdown` is rendered server-side from the analyst's structured `AnalystReport`:

- a headline;
- optional deterministic `coverage_notices` blockquotes;
- `#### Key Metrics (deterministic engine)` with `period_of_report` dates and inline clickable XBRL tags;
- `#### Findings`, each backed by verified `chunk_id`(s) and a verbatim `*Evidence:*` quote.

There is **no uncited `Notes` section** (`notes` was removed from `AnalystReport`). Clarification (`clarification_needed`) and out-of-corpus (`out_of_corpus`) turns display plain-English explanations plus clickable suggested follow-up chips. If all findings fail verification (even after the single verification-guided regeneration attempt), the answer is shown as **Answer Withheld**.

There is **no offline/mock fallback** in `edgarApi.ts`; API failures propagate as explicit error states.

## Authentication

`App.tsx` reads `GET /api/v1/auth/config` and picks a sign-in path:

1. **IAP session (Cloud Run with IAP):** when `iap.session_auth_enabled` is set, the app probes `/api/v1/auth/me` without a token. IAP's signed `X-Goog-IAP-JWT-Assertion` header signs the user in silently. Sign Out also clears the IAP session cookie.
2. **Workspace / Okta OIDC:** redirects to the IdP, reads the ID token from the URL fragment, and exchanges it at `/api/v1/auth/sso/exchange`. The bearer JWT is stored and attached to every API call.
3. **Local development:** outside Cloud Run (`IAP_STRICT_MODE=false`), `POST /api/v1/auth/sso/dev-token` mints a local ES256 developer JWT.

A 403 from the backend's domain or RBAC policy is shown in the gate panel.

## Tests

- `src/__tests__/App.test.tsx`: workspace shell, SSE streaming progress & result rendering, peer-comparison query with variance math, inline XBRL & text citation chips, `XBRL Fact` inspector pane, coverage notices / suggested follow-up chips, and `/` keyboard shortcut with `<GO>`.
- `src/__tests__/Auth.test.tsx`: SSO gate with bearer JWTs and sign-out, dev vs. strict OIDC configuration, direct IAP/OIDC bearer verification with a 403 domain error, and silent IAP-session sign-in.
- `src/__tests__/GroundingFactCards.test.tsx`: `VarianceTableCard` (`period_of_report` dates, `DIFFERING FY-END DATES` badge, `[XBRL]` provenance triggers), `MaterialFlagsCard`, and `DisclosureDeltaCard` rendering.

Total: **14 Vitest tests** (`cd frontend && npm test -- --run`).
