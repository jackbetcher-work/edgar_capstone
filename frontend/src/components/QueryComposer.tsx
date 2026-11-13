import { useEffect, useRef, type FormEvent, type KeyboardEvent } from 'react'
import { Loader2 } from 'lucide-react'
import type { SearchScopeFilters } from '../types/edgar'

interface QueryComposerProps {
  value: string
  onChange: (val: string) => void
  onSubmit: () => void
  isSubmitting: boolean
  filters: SearchScopeFilters
}

export function QueryComposer({
  value,
  onChange,
  onSubmit,
  isSubmitting,
  filters,
}: QueryComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 128)}px`
  }, [value])

  // Press '/' anywhere outside inputs to focus the terminal command prompt
  useEffect(() => {
    function handleGlobalSlash(e: globalThis.KeyboardEvent) {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return
      const activeTag = (document.activeElement?.tagName || '').toLowerCase()
      if (
        activeTag === 'input' ||
        activeTag === 'textarea' ||
        activeTag === 'select'
      ) {
        return
      }
      e.preventDefault()
      textareaRef.current?.focus()
    }
    window.addEventListener('keydown', handleGlobalSlash)
    return () => window.removeEventListener('keydown', handleGlobalSlash)
  }, [])

  const handleFormSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (!value.trim() || isSubmitting) return
    onSubmit()
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (value.trim() && !isSubmitting) {
        onSubmit()
      }
    }
  }

  const scopeSummary = [
    filters.selectedTickers.length > 0
      ? filters.selectedTickers.join(',')
      : filters.gicsSector !== 'ALL'
        ? filters.gicsSector.toUpperCase()
        : 'AUTO EQTY',
    `FY${String(filters.startYear).slice(-2)}–FY${String(filters.endYear).slice(-2)}`,
    filters.sectionId === 'ALL'
      ? '1A+7'
      : filters.sectionId === 'item_1a'
        ? '1A'
        : '7',
  ].join(' | ')

  return (
    <div className="surface-header border-t border-hairline px-3 py-2 shrink-0">
      <form
        id="analyst-query-form"
        onSubmit={handleFormSubmit}
        className="flex items-end gap-2 surface-input border border-strong px-2.5 py-1.5 focus-within:border-[var(--bb-amber)] transition-colors"
      >
        <span
          aria-hidden="true"
          className="tabular-nums-mono text-xs font-bold text-bb-amber pb-1 shrink-0 select-none"
        >
          EDGR&gt;
        </span>

        <label htmlFor="analyst-query-textarea" className="sr-only">
          Ask a natural language financial or SEC 10-K question
        </label>
        <textarea
          id="analyst-query-textarea"
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isSubmitting}
          placeholder="Enter variance query, peer comparison, or 10-K risk prompt... [Press / to focus]"
          className="flex-1 resize-none bg-transparent py-0.5 text-xs text-ink-primary placeholder:text-[var(--text-muted)] focus:outline-none leading-relaxed max-h-32 tabular-nums-mono"
        />

        <span className="hidden md:inline-block tabular-nums-mono text-[10px] text-ink-muted pb-1 shrink-0">
          {scopeSummary}
        </span>

        <button
          id="btn-submit-analyst-query"
          type="submit"
          disabled={!value.trim() || isSubmitting}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] btn-terminal-go disabled:opacity-35 cursor-pointer disabled:cursor-not-allowed shrink-0"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="w-3 h-3 animate-spin" />
              <span>EXEC</span>
            </>
          ) : (
            <span>&lt;GO&gt;</span>
          )}
        </button>
      </form>
    </div>
  )
}
