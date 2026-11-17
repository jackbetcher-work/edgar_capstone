import { useState } from 'react'
import {
  AlertTriangle,
  Check,
  Copy,
  HelpCircle,
  Info,
  Layers,
  Loader2,
  ShieldCheck,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
} from 'lucide-react'
import type {
  ConversationTurn,
  GroundedCitation,
  XbrlFactSelection,
} from '../types/edgar'
import {
  getIntentDisplayMeta,
  type ParsedInlineCitationToken,
  type ParsedInlineXbrlToken,
} from '../utils/formatters'
import { DisclosureDeltaCard } from './DisclosureDeltaCard'
import { GroundedMarkdown } from './GroundedMarkdown'
import { MaterialFlagsCard } from './MaterialFlagsCard'
import { VarianceTableCard } from './VarianceTableCard'

interface ChatTurnItemProps {
  turn: ConversationTurn
  activeChunkId: string | null
  onSelectInlineCitation: (
    token: ParsedInlineCitationToken,
    turnCitations: GroundedCitation[]
  ) => void
  onSelectXbrlCitation?: (
    token: ParsedInlineXbrlToken,
    turn: ConversationTurn
  ) => void
  onInspectXbrlFact?: (fact: XbrlFactSelection) => void
  onSelectFollowup?: (query: string) => void
  onFocusTurnCitations: (citations: GroundedCitation[]) => void
  onInspectFilingSection: (
    ticker: string,
    fiscalYear: number,
    sectionId: 'item_7' | 'item_1a'
  ) => void
  onSubmitFeedback: (
    turnId: string,
    rating: 'up' | 'down',
    comment: string
  ) => void
}

function computeProgressPercent(meta: Record<string, unknown> | null | undefined): number | null {
  if (!meta) return null
  const num = (k: string): number | null => {
    const v = meta[k]
    return typeof v === 'number' && Number.isFinite(v) ? v : null
  }
  const completed =
    num('companies_analyzed') ??
    num('completed_batches') ??
    num('completed') ??
    null
  const total =
    num('total_companies') ??
    num('total_batches') ??
    num('total') ??
    null
  if (completed !== null && total !== null && total > 0) {
    return Math.max(0, Math.min(100, Math.round((completed / total) * 100)))
  }
  return null
}

export function ChatTurnItem({
  turn,
  activeChunkId,
  onSelectInlineCitation,
  onSelectXbrlCitation,
  onInspectXbrlFact,
  onSelectFollowup,
  onFocusTurnCitations,
  onInspectFilingSection,
  onSubmitFeedback,
}: ChatTurnItemProps) {
  const [copied, setCopied] = useState(false)
  const [showCommentInput, setShowCommentInput] = useState(false)
  const [commentDraft, setCommentDraft] = useState('')

  const handleCopy = async () => {
    if (!turn.response) return
    try {
      await navigator.clipboard.writeText(turn.response.answer_markdown)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      // Clipboard API unavailable in restricted contexts
    }
  }

  const resp = turn.response
  const intentMeta = resp ? getIntentDisplayMeta(resp.intent) : null

  return (
    <article
      id={`conversation-turn-${turn.id}`}
      className="border border-hairline surface-panel mb-3"
    >
      {/* Terminal Command Input Echo Line */}
      <div className="px-3 py-1.5 surface-header border-b border-hairline flex flex-wrap items-center justify-between gap-2 tabular-nums-mono">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[11px] font-bold text-bb-amber shrink-0">
            CMD&gt;
          </span>
          <span className="text-xs font-medium text-ink-primary break-words">
            {turn.userQuery}
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-[10px] text-ink-muted uppercase shrink-0">
          <span>
            FY{String(turn.appliedFilters.startYear).slice(-2)}–FY
            {String(turn.appliedFilters.endYear).slice(-2)}
          </span>
          {turn.appliedFilters.selectedTickers.length > 0 && (
            <>
              <span>|</span>
              <span className="text-bb-yellow font-semibold">
                {turn.appliedFilters.selectedTickers.join(',')}
              </span>
            </>
          )}
          {turn.appliedFilters.sectionId !== 'ALL' && (
            <>
              <span>|</span>
              <span>
                {turn.appliedFilters.sectionId === 'item_1a'
                  ? 'ITEM 1A'
                  : 'ITEM 7'}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Loading State with Live SSE Stream Progress */}
      {turn.isLoading && (() => {
        const pct = computeProgressPercent(turn.progressMeta)
        const elapsedLabel =
          typeof turn.progressElapsedMs === 'number' && turn.progressElapsedMs > 0
            ? `${(turn.progressElapsedMs / 1000).toFixed(1)}s`
            : null
        return (
          <div
            id={`loading-turn-${turn.id}`}
            className="p-3.5 space-y-2.5 tabular-nums-mono"
          >
            <div className="flex items-center justify-between gap-2 text-xs text-bb-amber">
              <div className="flex items-center gap-2 min-w-0">
                <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                <span className="truncate">
                  {turn.progressMessage ||
                    'COMPUTING XBRL VARIANCES & RETRIEVING 10-K FILINGS...'}
                </span>
              </div>
              <div className="flex items-center gap-2 text-[10px] text-ink-muted shrink-0">
                {pct !== null && (
                  <span className="text-bb-yellow font-semibold">{pct}%</span>
                )}
                {elapsedLabel && <span>{elapsedLabel}</span>}
              </div>
            </div>
            {pct !== null ? (
              <div className="w-full h-1.5 surface-subtle overflow-hidden border border-hairline">
                <div
                  className="h-full bg-[var(--bb-amber)] transition-all duration-300"
                  style={{ width: `${pct}%` }}
                />
              </div>
            ) : (
              <div className="space-y-1.5">
                <div className="h-2.5 surface-subtle w-3/4 animate-pulse" />
                <div className="h-2.5 surface-subtle w-full animate-pulse" />
                <div className="h-2.5 surface-subtle w-2/3 animate-pulse" />
              </div>
            )}
          </div>
        )
      })()}

      {/* Error State */}
      {turn.errorMessage && !turn.isLoading && (
        <div
          role="alert"
          className="m-3 p-2.5 badge-negative text-xs flex items-center gap-2 tabular-nums-mono">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span>{turn.errorMessage}</span>
        </div>
      )}

      {/* Grounded Terminal Dispatch Body */}
      {resp && !turn.isLoading && intentMeta && (
        <div className="p-3">
          {/* Status & Action Strip */}
          <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 mb-3 border-b border-hairline tabular-nums-mono text-[11px]">
            <div className="flex flex-wrap items-center gap-2">
              <span
                className={`px-1.5 py-0.2 text-[10px] font-bold uppercase badge-${intentMeta.tone}`}
              >
                {intentMeta.shortCode}
              </span>
              <span className="text-ink-secondary text-[11px]">
                {intentMeta.label}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-1.5">
              {resp.model_armor_blocked ? (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.2 text-[10px] uppercase font-semibold badge-negative">
                  <AlertTriangle className="w-3 h-3" />
                  Guardrail Enforced
                </span>
              ) : resp.status === 'clarification' ? (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.2 text-[10px] uppercase font-semibold badge-amber">
                  <HelpCircle className="w-3 h-3" />
                  Clarification Needed
                </span>
              ) : resp.status === 'out_of_corpus' ? (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.2 text-[10px] uppercase font-semibold badge-amber">
                  <Info className="w-3 h-3" />
                  Outside Corpus Scope
                </span>
              ) : resp.answer_verified === false ? (
                <span
                  className="inline-flex items-center gap-1 px-1.5 py-0.2 text-[10px] uppercase font-semibold badge-negative"
                  title="The live answer failed verification, so its narrative was withheld"
                >
                  <AlertTriangle className="w-3 h-3" />
                  Answer Withheld
                </span>
              ) : resp.fact_check.passed ? (
                <span
                  className="inline-flex items-center gap-1 px-1.5 py-0.2 text-[10px] uppercase font-semibold badge-positive"
                  title="Numeric & citation fidelity verified against XBRL & 10-K corpus"
                >
                  <ShieldCheck className="w-3 h-3" />
                  Fact-Check Verified
                </span>
              ) : (
                <span
                  className="inline-flex items-center gap-1 px-1.5 py-0.2 text-[10px] uppercase font-semibold badge-amber"
                  title={resp.fact_check.discrepancies.join(' ')}
                >
                  <AlertTriangle className="w-3 h-3" />
                  Review Citations
                </span>
              )}

              {resp.sector_coverage &&
                typeof resp.sector_coverage.total_sector_companies === 'number' &&
                resp.sector_coverage.total_sector_companies > 0 && (
                  <span
                    className="inline-flex items-center gap-1 px-1.5 py-0.2 text-[10px] uppercase font-semibold badge-accent"
                    title={`Full-sector batched map-reduce: ${resp.sector_coverage.companies_analyzed ?? 0} of ${resp.sector_coverage.total_sector_companies} issuers analyzed across ${resp.sector_coverage.batches_executed ?? 0} parallel batches`}
                  >
                    <Layers className="w-3 h-3" />
                    {resp.sector_coverage.companies_analyzed ?? 0}/
                    {resp.sector_coverage.total_sector_companies}{' '}
                    {resp.sector_coverage.gics_sector || 'SECTOR'} ISSUERS
                    {typeof resp.sector_coverage.companies_with_matches === 'number'
                      ? ` · ${resp.sector_coverage.companies_with_matches} MATCHED`
                      : ''}
                  </span>
                )}

              {resp.context_and_memory?.turn_context_latency && (
                <span
                  className="px-1.5 py-0.2 text-[10px] uppercase text-ink-secondary border border-hairline"
                  title={`Reasoning: ${resp.context_and_memory.models.financial_reasoning_model} | Eval/Coding: ${resp.context_and_memory.models.coding_and_evaluation_model} | Cached Tokens: ${resp.context_and_memory.turn_context_latency.total_cached_context_tokens}`}
                >
                  ReAct:{resp.react_execution?.iterations ?? resp.agent_trace.length} · CTX:{resp.context_and_memory.turn_context_latency.mean_context_latency_ms.toFixed(1)}ms
                </span>
              )}

              {resp.citations.length > 0 && (
                <button
                  id={`btn-focus-citations-${turn.id}`}
                  type="button"
                  onClick={() => onFocusTurnCitations(resp.citations)}
                  className="px-1.5 py-0.2 text-[10px] uppercase btn-terminal-key cursor-pointer"
                >
                  [{resp.citations.length} SOURCES]
                </button>
              )}

              <button
                id={`btn-copy-report-${turn.id}`}
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center gap-1 px-1.5 py-0.2 text-[10px] uppercase btn-terminal-key cursor-pointer"
                title="Copy report markdown"
              >
                {copied ? (
                  <>
                    <Check className="w-3 h-3 text-bb-pos" />
                    <span>COPIED</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    <span>COPY</span>
                  </>
                )}
              </button>

              <div className="inline-flex items-center gap-0.5 ml-0.5">
                <button
                  id={`btn-feedback-up-${turn.id}`}
                  type="button"
                  onClick={() => {
                    onSubmitFeedback(turn.id, 'up', turn.feedbackComment || '')
                  }}
                  aria-label="Approve grounded synthesis"
                  className={`p-1 cursor-pointer transition-colors ${
                    turn.feedbackRating === 'up'
                      ? 'badge-positive'
                      : 'btn-terminal-key'
                  }`}
                >
                  <ThumbsUp className="w-3 h-3" />
                </button>
                <button
                  id={`btn-feedback-down-${turn.id}`}
                  type="button"
                  onClick={() => {
                    setShowCommentInput(true)
                    onSubmitFeedback(turn.id, 'down', turn.feedbackComment || '')
                  }}
                  aria-label="Flag synthesis issue"
                  className={`p-1 cursor-pointer transition-colors ${
                    turn.feedbackRating === 'down'
                      ? 'badge-negative'
                      : 'btn-terminal-key'
                  }`}
                >
                  <ThumbsDown className="w-3 h-3" />
                </button>
              </div>
            </div>
          </div>

          {/* Deterministic Data Coverage Notices */}
          {resp.coverage_notices && resp.coverage_notices.length > 0 && (
            <div
              id={`coverage-notices-${turn.id}`}
              className="mb-3 space-y-1.5"
              aria-label="Filing Coverage Notices"
            >
              {resp.coverage_notices.map((notice, idx) => (
                <div
                  key={`${turn.id}-notice-${notice.code}-${idx}`}
                  className={`px-2.5 py-1.5 text-[11.5px] leading-snug border flex items-start gap-2 ${
                    notice.severity === 'warning'
                      ? 'badge-amber'
                      : 'surface-subtle border-hairline text-ink-secondary'
                  }`}
                >
                  {notice.severity === 'warning' ? (
                    <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                  ) : (
                    <Info className="w-3.5 h-3.5 shrink-0 mt-0.5 text-bb-cyan" />
                  )}
                  <span className="font-sans">{notice.message}</span>
                </div>
              ))}
            </div>
          )}

          {/* Deterministic Variance Table */}
          {resp.variance_reports.length > 0 && (
            <VarianceTableCard
              turnId={turn.id}
              reports={resp.variance_reports}
              onInspectFilingSection={onInspectFilingSection}
              onInspectXbrlFact={onInspectXbrlFact}
            />
          )}

          {/* Deterministic Anomaly Flags */}
          {resp.material_variance_flags && resp.material_variance_flags.length > 0 && (
            <MaterialFlagsCard
              turnId={turn.id}
              flags={resp.material_variance_flags}
              totalFlags={resp.material_variance_summary?.total_flags}
              citations={resp.citations}
              onSelectCitation={(token) => onSelectInlineCitation(token, resp.citations)}
              onInspectXbrlFact={onInspectXbrlFact}
            />
          )}

          {/* Year-over-Year Disclosure Deltas (thematic tracking) */}
          {resp.disclosure_deltas && (
            <DisclosureDeltaCard
              turnId={turn.id}
              result={resp.disclosure_deltas}
              citations={resp.citations}
              onSelectCitation={(token) => onSelectInlineCitation(token, resp.citations)}
            />
          )}

          {/* Grounded Markdown Narrative with Interactive Inline Citations */}
          <GroundedMarkdown
            turnId={turn.id}
            markdown={resp.answer_markdown}
            activeChunkId={activeChunkId}
            onSelectCitation={(token) =>
              onSelectInlineCitation(token, resp.citations)
            }
            onSelectXbrlCitation={
              onSelectXbrlCitation
                ? (token) => onSelectXbrlCitation(token, turn)
                : undefined
            }
          />

          {/* Interactive Suggested Follow-Up Chips (Clarification / Out-of-Corpus) */}
          {resp.suggested_followups && resp.suggested_followups.length > 0 && (
            <div
              id={`suggested-followups-${turn.id}`}
              className="mt-3 pt-2.5 border-t border-hairline"
            >
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-ink-muted tabular-nums-mono mb-1.5">
                <Sparkles className="w-3 h-3 text-bb-amber" />
                <span>Try a grounded follow-up query:</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {resp.suggested_followups.map((suggestion, idx) => (
                  <button
                    key={`${turn.id}-sug-${idx}`}
                    id={`btn-followup-${turn.id}-${idx}`}
                    type="button"
                    onClick={() => onSelectFollowup?.(suggestion)}
                    className="px-2 py-1 text-[11px] font-sans text-left btn-terminal-key hover:border-[var(--bb-amber)] cursor-pointer transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {showCommentInput && (
            <div className="mt-3 pt-2.5 border-t border-hairline flex items-center gap-2 tabular-nums-mono">
              <input
                id={`input-feedback-note-${turn.id}`}
                type="text"
                value={commentDraft}
                onChange={(e) => setCommentDraft(e.target.value)}
                placeholder="ENTER AUDIT NOTE..."
                className="flex-1 px-2 py-1 text-xs surface-input border border-strong text-ink-primary focus:outline-none focus:border-[var(--bb-amber)]"
              />
              <button
                id={`btn-save-feedback-note-${turn.id}`}
                type="button"
                onClick={() => {
                  onSubmitFeedback(
                    turn.id,
                    turn.feedbackRating || 'down',
                    commentDraft
                  )
                  setShowCommentInput(false)
                }}
                className="px-2.5 py-1 text-[11px] btn-terminal-go cursor-pointer"
              >
                LOG
              </button>
            </div>
          )}
        </div>
      )}
    </article>
  )
}
