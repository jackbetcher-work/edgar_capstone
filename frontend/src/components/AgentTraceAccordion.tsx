import { useState } from 'react'
import { CheckCircle2, ChevronDown, ChevronRight, Cpu, ShieldAlert } from 'lucide-react'
import type { AgentTraceStep } from '../types/edgar'

interface AgentTraceAccordionProps {
  turnId: string
  steps: AgentTraceStep[]
  totalLatencyMs: number
  candidatesSearched?: number
}

export function AgentTraceAccordion({
  turnId,
  steps,
  totalLatencyMs,
  candidatesSearched,
}: AgentTraceAccordionProps) {
  const [isOpen, setIsOpen] = useState(false)

  if (!steps || steps.length === 0) return null

  return (
    <div className="mb-3 rounded-lg surface-subtle border border-hairline overflow-hidden">
      <button
        id={`btn-trace-toggle-${turnId}`}
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
        className="w-full px-3 py-2 flex items-center justify-between gap-2 text-left text-xs text-ink-secondary hover:text-ink-primary cursor-pointer transition-colors"
      >
        <div className="flex items-center gap-2 min-w-0">
          {isOpen ? (
            <ChevronDown className="w-3.5 h-3.5 shrink-0 text-ink-muted" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 shrink-0 text-ink-muted" />
          )}
          <Cpu className="w-3.5 h-3.5 shrink-0 text-ink-muted" aria-hidden="true" />
          <span className="font-medium truncate">
            ReAct Orchestration Trace ({steps.length} tool stages)
          </span>
          {candidatesSearched !== undefined && candidatesSearched > 0 && (
            <span className="hidden sm:inline text-ink-muted">
              · {candidatesSearched.toLocaleString()} chunks evaluated
            </span>
          )}
        </div>
        <span className="tabular-nums-mono text-[11px] text-ink-muted shrink-0">
          {totalLatencyMs.toFixed(1)} ms
        </span>
      </button>

      {isOpen && (
        <div className="border-t border-hairline px-3 py-2.5 space-y-2 surface-panel">
          {steps.map((step) => (
            <div
              key={`${step.step}-${step.tool}`}
              className="flex items-start justify-between gap-3 text-xs"
            >
              <div className="flex items-start gap-2 min-w-0">
                {step.status === 'blocked' ? (
                  <ShieldAlert
                    className="w-3.5 h-3.5 mt-0.5 shrink-0"
                    style={{ color: 'var(--negative-ink)' }}
                  />
                ) : (
                  <CheckCircle2
                    className="w-3.5 h-3.5 mt-0.5 shrink-0"
                    style={{ color: 'var(--positive-ink)' }}
                  />
                )}
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="font-semibold text-ink-primary">
                      {step.agent}
                    </span>
                    <span className="text-ink-muted">→</span>
                    <code className="tabular-nums-mono text-[11px] px-1.5 py-0.2 rounded surface-subtle text-ink-secondary">
                      {step.tool}
                    </code>
                  </div>
                  <p className="text-ink-secondary text-[11px] mt-0.5 leading-relaxed">
                    {step.summary}
                  </p>
                </div>
              </div>
              <span className="tabular-nums-mono text-[11px] text-ink-muted shrink-0">
                {step.duration_ms.toFixed(1)} ms
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
