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
| `npm test` | Vitest + Testing Library in jsdom (`src/__tests__/`) |

## Layout

```text
src/
├── App.tsx                      # Workspace state: auth/session bootstrap, chat turns, feedback, source pane (the backend resolves scope from the question)
├── components/
│   ├── AuthGatePanel.tsx        # Sign-in gate (IAP session, Workspace/Okta OIDC, local dev token)
│   ├── HeaderBar.tsx            # Signed-in identity, role badge, sign-out
│   ├── StarterPrompts.tsx       # Example longitudinal, peer-comparison, and thematic queries
│   ├── QueryComposer.tsx        # EDGR> natural-language command line ("/" to focus, <GO> to run); placeholder shows corpus coverage
│   ├── ChatTurnItem.tsx         # One answer: interpreted-scope echo (from `resolved_scope`), live SSE progress, coverage notices, follow-up chips, rendered report, collapsible variance/flag/delta cards
│   ├── EvidencePanel.tsx        # Collapsible frame with digest header for the supporting-evidence cards
│   ├── FilingMarkdown.tsx       # Structured rendering of 10-K corpus chunks (headings, bullets, repaired SEC tables)
│   ├── DisclosureDeltaCard.tsx  # Deterministic YoY wording deltas (added / removed / modified passages)
│   ├── GroundedMarkdown.tsx     # Markdown subset (###/#### headings, blockquotes, bullets, bold/code/italic) + clickable inline text & XBRL citation chips
│   ├── MaterialFlagsCard.tsx    # Deterministic anomaly flags with period_of_report, xbrl_concept & clickable [XBRL] triggers
│   ├── VarianceTableCard.tsx    # Deterministic variance table per company with period_of_report dates, DIFFERING FY-END DATES badge & [XBRL] triggers
│   ├── SourceInspectorPane.tsx  # Right pane shell hosting the 3 modular tabs under `inspector/`
│   └── inspector/
│       ├── CitationsTab.tsx     # De-duplicated, collapsible cited passages with section filter, expand-all and per-filing variants
│       ├── FilingReaderTab.tsx  # Full 10-K Markdown section viewer with keyword search & highlight
│       └── XbrlFactTab.tsx      # XBRL provenance inspector (accession, concept, derived calculation inputs, MD&A jump)
├── services/
│   ├── httpClient.ts            # Shared authenticated fetch, `ApiAuthError` (401/403), and FastAPI/GCP JSON error extraction
│   ├── authService.ts           # Bearer token storage, /api/v1/auth/* calls, OIDC fragment handling
│   └── edgarApi.ts              # SSE streaming (/api/v1/chat/stream), /companies, /section, /feedback
├── types/edgar.ts               # API response, SSE progress, NumericCitation, CoverageNotice & XBRL types
└── utils/
    ├── citationDedup.ts         # Collapses repeated / cross-year duplicate passages and maps aliases to the kept card
    ├── citationTokens.ts        # Builds canonical inline-citation tokens for the source pane
    ├── deterministicEvidence.ts # Render predicates for variance, anomaly-flag, and disclosure-delta cards
    ├── filingMarkdown.ts        # Parses corpus chunk Markdown (strips header/escapes/footers, repairs SEC tables)
    ├── formatters.ts            # USD / percent / bps / period-end date formatting, combined text+XBRL citation parser & status badges
    ├── resolvedScope.ts         # Formats the backend's `resolved_scope` for the per-answer scope echo
    ├── xbrlLabels.ts            # Human labels for XBRL concepts and 10-K section breadcrumbs (citation chips)
    └── xbrlFactSelection.ts     # Maps variance rows, anomaly flags, and inline XBRL chips to `XbrlFactSelection` records
```

## How answers are rendered

The UI streams turns over **`POST /api/v1/chat/stream`** (SSE) with live progress events (`connected`, `scope_resolved`, `sector_map_progress`, `synthesizing`, `result`, `error`).

Responses return `answer_markdown` containing:
- **Inline text citation tags:** `[TICKER FYxxxx | Section Title | chunk_id](sec_filing_url)` — opens the **Cited Chunk** / **Full Section** view in `SourceInspectorPane`.
- **Inline XBRL numeric citation tags:** `[TICKER FYxxxx | XBRL <concept> | acc:<accession_number>](sec_filing_url)` — opens the **XBRL Fact** tab in `SourceInspectorPane` showing accession numbers, XBRL concepts, denominator concepts, derived calculation inputs, SEC EDGAR filing links, and an MD&A jump button.
- **Deterministic blockquotes (`coverage_notices`):** explicit notices for missing fiscal years (e.g., calendar-year filers whose FY2026 10-Ks file in Q1 2027), fiscal-calendar misalignment (`DIFFERING FY-END DATES`), and full-sector map-reduce coverage counts.

For verified live answers, `answer_markdown` is rendered server-side from the analyst's structured `AnalystReport`:

- a headline;
- optional deterministic `coverage_notices` blockquotes;
- `#### Key Metrics (deterministic engine)` with `period_of_report` dates and inline clickable XBRL tags;
- `#### Findings`, each backed by verified `chunk_id`(s) and a verbatim `*Evidence:*` quote.

Clarification (`clarification_needed`) and out-of-corpus (`out_of_corpus`) turns display plain-English explanations plus clickable suggested follow-up chips. If all findings fail verification (even after the single verification-guided regeneration attempt), the answer is shown as **Answer Withheld**.

API requests target the live backend directly, and backend errors surface as explicit states in the UI.

## Authentication

`App.tsx` reads `GET /api/v1/auth/config` and picks a sign-in path:

1. **IAP session (Cloud Run with IAP):** when `iap.session_auth_enabled` is set, the app probes `/api/v1/auth/me` without a token. IAP's signed `X-Goog-IAP-JWT-Assertion` header signs the user in silently. Sign Out also clears the IAP session cookie.
2. **Workspace / Okta OIDC:** redirects to the IdP, reads the ID token from the URL fragment, and exchanges it at `/api/v1/auth/sso/exchange`. The bearer JWT is stored and attached to every API call.
3. **Local development:** outside Cloud Run (`IAP_STRICT_MODE=false`), `POST /api/v1/auth/sso/dev-token` mints a local ES256 developer JWT.

A 403 from the backend's domain or RBAC policy is shown in the gate panel.

## Tests

Vitest + Testing Library suite under `src/__tests__/` (`cd frontend && npm test -- --run`):

- `App.test.tsx`: workspace shell, SSE streaming progress & result rendering, peer-comparison query with variance math, inline XBRL & text citation chips, `XBRL Fact` inspector pane, coverage notices / suggested follow-up chips, and `/` keyboard shortcut with `<GO>`.
- `Auth.test.tsx`: SSO gate with bearer JWTs and sign-out, dev vs. strict OIDC configuration, direct IAP/OIDC bearer verification with a 403 domain error, and silent IAP-session sign-in.
- `ChatTurnItem.test.tsx`: per-turn rendering, selective display of variance/flag/delta cards, coverage notices, and withheld-answer states.
- `Components.test.tsx`: `GroundedMarkdown`, `VarianceTableCard`, and `FilingReaderTab` rendering and interaction.
- `EvidenceUi.test.tsx`: `EvidencePanel`, `CitationsTab`, and inline table/excerpt rendering.
- `GroundingFactCards.test.tsx`: `MaterialFlagsCard` and `DisclosureDeltaCard` rendering and `[XBRL]` provenance triggers.
- `resolvedScope.test.tsx`: `resolved_scope` formatting and per-answer scope echo rendering.
- `filingEvidence.test.ts`: citation deduplication, corpus chunk Markdown parsing, and XBRL/section label helpers.
- `services.test.ts`: `httpClient`, `authService`, and `edgarApi` SSE streaming and error handling.
- `utils.test.ts`: formatting helpers and `xbrlFactSelection` builders.
