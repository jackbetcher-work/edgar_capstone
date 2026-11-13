import { useState } from 'react'
import {
  AlertTriangle,
  Check,
  Copy,
  Loader2,
  ShieldCheck,
  ThumbsDown,
  ThumbsUp,
} from 'lucide-react'
import type { ConversationTurn, GroundedCitation } from '../types/edgar'
import {
  getIntentDisplayMeta,
  type ParsedInlineCitationToken,
} from '../utils/formatters'
import { GroundedMarkdown } from './GroundedMarkdown'
import { VarianceTableCard } from './VarianceTableCard'

interface ChatTurnItemProps {
  turn: ConversationTurn
  activeChunkId: string | null
  onSelectInlineCitation: (
    token: ParsedInlineCitationToken,
    turnCitations: GroundedCitation[]
  ) => void
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

export function ChatTurnItem({
  turn,
  activeChunkId,
  onSelectInlineCitation,
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

      {/* Loading State */}
      {turn.isLoading && (
        <div
          id={`loading-turn-${turn.id}`}
          className="p-3.5 space-y-2.5 tabular-nums-mono"
        >
          <div className="flex items-center gap-2 text-xs text-bb-amber">
            <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
            <span>COMPUTING XBRL VARIANCES &amp; RETRIEVING 10-K FILINGS...</span>
          </div>
          <div className="space-y-1.5">
            <div className="h-2.5 surface-subtle w-3/4 animate-pulse" />
            <div className="h-2.5 surface-subtle w-full animate-pulse" />
            <div className="h-2.5 surface-subtle w-2/3 animate-pulse" />
          </div>
        </div>
      )}

      {/* Error State */}
      {turn.errorMessage && !turn.isLoading && (
        <div className="m-3 p-2.5 badge-negative text-xs flex items-center gap-2 tabular-nums-mono">
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

          {/* Deterministic Variance Table */}
          {resp.variance_reports.length > 0 && (
            <VarianceTableCard
              turnId={turn.id}
              reports={resp.variance_reports}
              onInspectFilingSection={onInspectFilingSection}
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
          />

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
