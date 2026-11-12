import { useEffect, useRef, type FormEvent, type KeyboardEvent } from 'react'
import { CornerDownLeft, Loader2, Sparkles } from 'lucide-react'
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

  // Auto-resize textarea height
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [value])

  // Press '/' anywhere outside inputs to focus the analyst prompt bar
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
      ? filters.selectedTickers.join(', ')
      : filters.gicsSector !== 'ALL'
        ? filters.gicsSector
        : 'Auto Entity Detection',
    `FY${filters.startYear}–FY${filters.endYear}`,
    filters.sectionId === 'ALL'
      ? 'Item 1A + Item 7'
      : filters.sectionId === 'item_1a'
        ? 'Item 1A Risk'
        : 'Item 7 MD&A',
  ].join(' · ')

  return (
    <div className="px-4 sm:px-6 pb-4 pt-2 shrink-0 bg-gradient-to-t from-[var(--bg-canvas)] via-[var(--bg-canvas)] to-transparent">
      <form
        id="analyst-query-form"
        onSubmit={handleFormSubmit}
        className="max-w-3xl mx-auto rounded-xl surface-composer border border-strong p-2.5 transition-all"
      >
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
          placeholder="Ask about multi-year revenue & margin variances, peer comparisons, or 10-K risk disclosures... (Press / to focus)"
          className="w-full resize-none bg-transparent px-2 py-1 text-sm text-ink-primary placeholder:text-[var(--text-muted)] focus:outline-none leading-relaxed max-h-40"
        />

        <div className="flex items-center justify-between gap-2 pt-2 px-1.5 border-t border-hairline mt-1">
          <div className="flex items-center gap-1.5 text-[11px] text-ink-muted tabular-nums-mono truncate">
            <Sparkles className="w-3 h-3 shrink-0" aria-hidden="true" />
            <span className="truncate">{scopeSummary}</span>
          </div>

          <button
            id="btn-submit-analyst-query"
            type="submit"
            disabled={!value.trim() || isSubmitting}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold btn-primary-ink disabled:opacity-40 cursor-pointer disabled:cursor-not-allowed shrink-0"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Synthesizing</span>
              </>
            ) : (
              <>
                <span>Run Analysis</span>
                <CornerDownLeft className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  )
}
