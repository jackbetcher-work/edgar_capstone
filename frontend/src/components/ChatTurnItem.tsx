import { useState } from 'react'
import {
  AlertTriangle,
  Check,
  Copy,
  FileSearch,
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
import { AgentTraceAccordion } from './AgentTraceAccordion'
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
      className="space-y-4 py-5 border-b border-hairline last:border-b-0 animate-turn-enter"
    >
      {/* Analyst Query Bubble */}
      <div className="flex flex-col items-end">
        <div className="max-w-2xl rounded-xl px-4 py-2.5 surface-subtle border border-hairline text-sm text-ink-primary leading-relaxed">
          {turn.userQuery}
        </div>
        <div className="flex items-center gap-2 mt-1 text-[11px] text-ink-muted tabular-nums-mono">
          <span>
            FY{turn.appliedFilters.startYear}–FY{turn.appliedFilters.endYear}
          </span>
          {turn.appliedFilters.selectedTickers.length > 0 && (
            <>
              <span>·</span>
              <span>{turn.appliedFilters.selectedTickers.join(', ')}</span>
            </>
          )}
          {turn.appliedFilters.sectionId !== 'ALL' && (
            <>
              <span>·</span>
              <span>
                {turn.appliedFilters.sectionId === 'item_1a'
                  ? 'Item 1A'
                  : 'Item 7'}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Loading State */}
      {turn.isLoading && (
        <div
          id={`loading-turn-${turn.id}`}
          className="rounded-lg surface-panel border border-hairline p-4 space-y-3"
        >
          <div className="flex items-center gap-2.5 text-xs font-medium text-ink-secondary">
            <Loader2
              className="w-4 h-4 animate-spin"
              style={{ color: 'var(--accent-ink)' }}
            />
            <span>
              RootOrchestrator executing deterministic variance math &amp; hybrid 10-K
              search...
            </span>
          </div>
          <div className="space-y-2 pt-1">
            <div className="h-3 rounded surface-subtle w-3/4 animate-pulse" />
            <div className="h-3 rounded surface-subtle w-full animate-pulse" />
            <div className="h-3 rounded surface-subtle w-2/3 animate-pulse" />
          </div>
        </div>
      )}

      {/* Error State */}
      {turn.errorMessage && !turn.isLoading && (
        <div className="rounded-lg badge-negative p-3.5 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{turn.errorMessage}</span>
        </div>
      )}

      {/* Grounded Agent Synthesis */}
      {resp && !turn.isLoading && intentMeta && (
        <div className="rounded-xl surface-panel border border-hairline p-4 sm:p-5">
          {/* Verification & Routing Header */}
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 mb-3 border-b border-hairline">
            <div className="flex flex-wrap items-center gap-2">
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-semibold tabular-nums-mono badge-${intentMeta.tone}`}
              >
                {intentMeta.shortCode}
              </span>
              <span className="text-xs font-medium text-ink-primary">
                {intentMeta.label}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-[11px] tabular-nums-mono">
              {resp.model_armor_blocked ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded badge-negative font-medium">
                  <AlertTriangle className="w-3 h-3" />
                  Guardrail Enforced
                </span>
              ) : resp.fact_check.passed ? (
                <span
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded badge-positive font-medium"
                  title="Post-generation fact-check confirmed numeric & citation fidelity"
                >
                  <ShieldCheck className="w-3 h-3" />
                  Fact-Check Verified
                </span>
              ) : (
                <span
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded badge-amber font-medium"
                  title={resp.fact_check.discrepancies.join(' ')}
                >
                  <AlertTriangle className="w-3 h-3" />
                  Review Citations
                </span>
              )}

              {resp.citations.length > 0 && (
                <button
                  id={`btn-focus-citations-${turn.id}`}
                  type="button"
                  onClick={() => onFocusTurnCitations(resp.citations)}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded surface-subtle border border-hairline text-ink-secondary hover:text-ink-primary cursor-pointer"
                >
                  <FileSearch className="w-3 h-3" />
                  {resp.citations.length} 10-K Sources
                </button>
              )}
            </div>
          </div>

          {/* ReAct Orchestration Trace */}
          <AgentTraceAccordion
            turnId={turn.id}
            steps={resp.agent_trace}
            totalLatencyMs={resp.total_latency_ms}
            candidatesSearched={resp.candidates_searched}
          />

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

          {/* Analyst Action & Audit Feedback Footer */}
          <div className="mt-4 pt-3 border-t border-hairline flex flex-wrap items-center justify-between gap-2 text-xs text-ink-muted">
            <div className="flex items-center gap-2">
              <button
                id={`btn-copy-report-${turn.id}`}
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center gap-1 px-2 py-1 rounded hover:bg-[var(--bg-subtle)] text-ink-secondary cursor-pointer transition-colors"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy Brief</span>
                  </>
                )}
              </button>

              {resp.context_cache && (
                <span className="hidden sm:inline text-[11px] tabular-nums-mono">
                  · Context Cache:{' '}
                  {resp.context_cache.cache_hit ? 'HIT' : 'WARMED'} (
                  {resp.context_cache.cache_key})
                </span>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              <span className="text-[11px] mr-1">Audit Rating:</span>
              <button
                id={`btn-feedback-up-${turn.id}`}
                type="button"
                onClick={() => {
                  onSubmitFeedback(turn.id, 'up', turn.feedbackComment || '')
                }}
                aria-label="Approve grounded synthesis"
                className={`p-1 rounded cursor-pointer transition-colors ${
                  turn.feedbackRating === 'up'
                    ? 'badge-positive'
                    : 'hover:bg-[var(--bg-subtle)] text-ink-secondary'
                }`}
              >
                <ThumbsUp className="w-3.5 h-3.5" />
              </button>
              <button
                id={`btn-feedback-down-${turn.id}`}
                type="button"
                onClick={() => {
                  setShowCommentInput(true)
                  onSubmitFeedback(turn.id, 'down', turn.feedbackComment || '')
                }}
                aria-label="Flag synthesis issue"
                className={`p-1 rounded cursor-pointer transition-colors ${
                  turn.feedbackRating === 'down'
                    ? 'badge-negative'
                    : 'hover:bg-[var(--bg-subtle)] text-ink-secondary'
                }`}
              >
                <ThumbsDown className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {showCommentInput && (
            <div className="mt-2.5 flex items-center gap-2">
              <input
                id={`input-feedback-note-${turn.id}`}
                type="text"
                value={commentDraft}
                onChange={(e) => setCommentDraft(e.target.value)}
                placeholder="Add optional audit note for compliance telemetry..."
                className="flex-1 px-2.5 py-1.5 rounded text-xs surface-subtle border border-hairline text-ink-primary focus:outline-none"
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
                className="px-2.5 py-1.5 rounded text-xs font-medium btn-primary-ink cursor-pointer"
              >
                Log Note
              </button>
            </div>
          )}
        </div>
      )}
    </article>
  )
}
