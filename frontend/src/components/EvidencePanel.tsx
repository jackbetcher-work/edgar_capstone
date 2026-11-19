import { useState, type ReactNode } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'

interface EvidencePanelProps {
  /** Accessible name of the region (stable; tests and screen readers rely on it). */
  ariaLabel: string
  title: string
  /** Compact one-line digest shown in the header, visible even when collapsed. */
  summary?: ReactNode
  /** Extra header badges (e.g. fiscal-calendar warnings). */
  badges?: ReactNode
  defaultOpen?: boolean
  toggleId?: string
  children: ReactNode
}

/**
 * Collapsible frame for supporting-evidence cards beneath an answer. The
 * header carries a digest so analysts can judge relevance before expanding.
 */
export function EvidencePanel({
  ariaLabel,
  title,
  summary,
  badges,
  defaultOpen = true,
  toggleId,
  children,
}: EvidencePanelProps) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section
      aria-label={ariaLabel}
      className="mb-2 border border-hairline surface-panel overflow-hidden"
    >
      <button
        id={toggleId}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className={`w-full px-3 py-1.5 surface-header flex flex-wrap items-center justify-between gap-2 tabular-nums-mono text-left cursor-pointer hover:bg-[var(--bg-elevated)] ${
          open ? 'border-b border-hairline' : ''
        }`}
      >
        <span className="inline-flex items-center gap-1.5 min-w-0">
          {open ? (
            <ChevronDown className="w-3 h-3 text-bb-amber shrink-0" aria-hidden="true" />
          ) : (
            <ChevronRight className="w-3 h-3 text-ink-muted shrink-0" aria-hidden="true" />
          )}
          <span className="text-[11px] font-semibold uppercase tracking-wider text-bb-amber">
            {title}
          </span>
        </span>
        <span className="inline-flex items-center gap-2 min-w-0">
          {badges}
          {summary && (
            <span className="text-[10px] text-ink-muted uppercase truncate">{summary}</span>
          )}
        </span>
      </button>
      {open && children}
    </section>
  )
}
