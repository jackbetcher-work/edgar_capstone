import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown, Search, X } from 'lucide-react'
import type {
  CompanySummary,
  SearchScopeFilters,
  SectionFilterId,
} from '../types/edgar'
import { formatUsdCompact } from '../utils/formatters'

interface HybridScopeBarProps {
  companies: CompanySummary[]
  sectors: string[]
  minYear: number
  maxYear: number
  filters: SearchScopeFilters
  onChangeFilters: (next: SearchScopeFilters) => void
}

const SECTION_OPTIONS: Array<{ id: SectionFilterId; label: string }> = [
  { id: 'ALL', label: 'ALL 10-K (1A + 7)' },
  { id: 'item_7', label: 'ITEM 7 · MD&A' },
  { id: 'item_1a', label: 'ITEM 1A · RISK' },
]

export function HybridScopeBar({
  companies,
  sectors,
  minYear,
  maxYear,
  filters,
  onChangeFilters,
}: HybridScopeBarProps) {
  const [isTickerMenuOpen, setIsTickerMenuOpen] = useState(false)
  const [tickerSearch, setTickerSearch] = useState('')
  const dropdownRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    function handleOutsideClick(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsTickerMenuOpen(false)
      }
    }
    if (isTickerMenuOpen) {
      document.addEventListener('mousedown', handleOutsideClick)
    }
    return () => document.removeEventListener('mousedown', handleOutsideClick)
  }, [isTickerMenuOpen])

  const years: number[] = []
  for (let yr = minYear; yr <= maxYear; yr += 1) {
    years.push(yr)
  }

  const filteredCompanies = companies.filter((c) => {
    if (!tickerSearch.trim()) return true
    const q = tickerSearch.trim().toLowerCase()
    return (
      c.ticker.toLowerCase().includes(q) ||
      c.company_name.toLowerCase().includes(q) ||
      c.gics_sector.toLowerCase().includes(q)
    )
  })

  const toggleTicker = (ticker: string) => {
    const exists = filters.selectedTickers.includes(ticker)
    const nextTickers = exists
      ? filters.selectedTickers.filter((t) => t !== ticker)
      : [...filters.selectedTickers, ticker]
    onChangeFilters({
      ...filters,
      selectedTickers: nextTickers,
    })
  }

  const hasCustomScope =
    filters.selectedTickers.length > 0 ||
    filters.gicsSector !== 'ALL' ||
    filters.sectionId !== 'ALL' ||
    filters.startYear !== 2024 ||
    filters.endYear !== 2026

  return (
    <nav
      id="hybrid-scope-bar"
      aria-label="Hybrid RAG Search Metadata Filters"
      className="surface-panel border-b border-hairline px-3 py-1.5 flex flex-wrap items-center gap-2 z-10 shrink-0 tabular-nums-mono"
    >
      <span className="text-[10px] font-bold uppercase tracking-wider text-bb-amber">
        SCOPE&gt;
      </span>

      {/* Multi-Issuer Equity Picker */}
      <div className="relative" ref={dropdownRef}>
        <button
          id="filter-issuer-trigger"
          type="button"
          onClick={() => setIsTickerMenuOpen((prev) => !prev)}
          aria-expanded={isTickerMenuOpen}
          className="inline-flex items-center gap-1.5 px-2 py-0.5 text-[11px] surface-input border border-hairline hover:border-strong cursor-pointer transition-colors"
        >
          <span className="text-ink-muted">EQTY:</span>
          <span className="text-bb-yellow font-semibold">
            {filters.selectedTickers.length === 0
              ? 'ALL S&P'
              : filters.selectedTickers.join(',')}
          </span>
          <ChevronDown className="w-3 h-3 text-ink-muted" aria-hidden="true" />
        </button>

        {isTickerMenuOpen && (
          <div
            id="filter-issuer-popover"
            className="absolute left-0 mt-1 w-88 surface-elevated border border-strong p-2 z-30"
          >
            <div className="relative mb-1.5">
              <Search
                className="w-3 h-3 text-ink-muted absolute left-2 top-1/2 -translate-y-1/2"
                aria-hidden="true"
              />
              <input
                id="input-issuer-search"
                type="text"
                value={tickerSearch}
                onChange={(e) => setTickerSearch(e.target.value)}
                placeholder="TICKER, ISSUER, OR SECTOR..."
                className="w-full pl-7 pr-2 py-1 text-[11px] surface-input border border-hairline text-ink-primary focus:outline-none focus:border-[var(--bb-amber)]"
              />
            </div>

            <div className="flex items-center justify-between px-1 pb-1 text-[10px] text-ink-muted border-b border-hairline mb-1 uppercase">
              <span>SELECT 1+ EQUITIES</span>
              {filters.selectedTickers.length > 0 && (
                <button
                  id="btn-clear-selected-issuers"
                  type="button"
                  onClick={() =>
                    onChangeFilters({ ...filters, selectedTickers: [] })
                  }
                  className="text-bb-amber hover:underline cursor-pointer"
                >
                  CLEAR ({filters.selectedTickers.length})
                </button>
              )}
            </div>

            <div className="max-h-64 overflow-y-auto divide-y divide-[var(--border-hairline)]">
              {filteredCompanies.map((company) => {
                const selected = filters.selectedTickers.includes(company.ticker)
                return (
                  <button
                    key={company.ticker}
                    id={`issuer-option-${company.ticker}`}
                    type="button"
                    onClick={() => toggleTicker(company.ticker)}
                    className={`w-full flex items-center justify-between px-2 py-1.5 text-left text-[11px] cursor-pointer transition-colors ${
                      selected
                        ? 'badge-amber'
                        : 'hover:bg-[var(--bg-subtle)] text-ink-primary'
                    }`}
                  >
                    <div className="min-w-0 pr-2">
                      <div className="flex items-center gap-2">
                        <span className="text-bb-yellow font-bold w-12 shrink-0">
                          {company.ticker}
                        </span>
                        <span className="truncate text-ink-primary">
                          {company.company_name}
                        </span>
                      </div>
                      <div className="text-[10px] text-ink-muted truncate">
                        {company.gics_sector} · REV{' '}
                        {formatUsdCompact(company.latest_revenue)} (FY
                        {String(company.latest_fiscal_year).slice(-2)})
                      </div>
                    </div>
                    {selected && <Check className="w-3.5 h-3.5 shrink-0" />}
                  </button>
                )
              })}
              {filteredCompanies.length === 0 && (
                <p className="text-[11px] text-ink-muted py-3 text-center">
                  NO MATCHING EQUITIES
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* GICS Sector Filter */}
      <div className="inline-flex items-center gap-1 px-2 py-0.5 surface-input border border-hairline text-[11px]">
        <label htmlFor="select-gics-sector" className="text-ink-muted">
          <span className="sr-only">GICS Sector</span>
          <span aria-hidden="true">SECT:</span>
        </label>
        <select
          id="select-gics-sector"
          aria-label="GICS Sector"
          value={filters.gicsSector}
          onChange={(e) =>
            onChangeFilters({ ...filters, gicsSector: e.target.value })
          }
          className="bg-transparent text-ink-primary font-medium focus:outline-none cursor-pointer"
        >
          <option value="ALL">ALL SECTORS</option>
          {sectors.map((sec) => (
            <option key={sec} value={sec}>
              {sec.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      {/* 10-K Section Filter */}
      <div className="inline-flex items-center gap-1 px-2 py-0.5 surface-input border border-hairline text-[11px]">
        <label htmlFor="select-10k-section" className="text-ink-muted">
          <span className="sr-only">Form 10-K Section</span>
          <span aria-hidden="true">DOC:</span>
        </label>
        <select
          id="select-10k-section"
          aria-label="Form 10-K Section"
          value={filters.sectionId}
          onChange={(e) =>
            onChangeFilters({
              ...filters,
              sectionId: e.target.value as SectionFilterId,
            })
          }
          className="bg-transparent text-ink-primary font-medium focus:outline-none cursor-pointer"
        >
          {SECTION_OPTIONS.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Fiscal Period Range */}
      <div className="inline-flex items-center gap-1 px-2 py-0.5 surface-input border border-hairline text-[11px]">
        <label htmlFor="select-start-year" className="text-ink-muted">
          FY:
        </label>
        <select
          id="select-start-year"
          value={filters.startYear}
          onChange={(e) => {
            const nextStart = Number(e.target.value)
            onChangeFilters({
              ...filters,
              startYear: nextStart,
              endYear: Math.max(nextStart, filters.endYear),
            })
          }}
          className="bg-transparent text-ink-primary font-semibold focus:outline-none cursor-pointer"
        >
          {years.map((yr) => (
            <option key={yr} value={yr}>
              {yr}
            </option>
          ))}
        </select>
        <span className="text-ink-muted">–</span>
        <label htmlFor="select-end-year" className="sr-only">
          End Fiscal Year
        </label>
        <select
          id="select-end-year"
          value={filters.endYear}
          onChange={(e) => {
            const nextEnd = Number(e.target.value)
            onChangeFilters({
              ...filters,
              startYear: Math.min(filters.startYear, nextEnd),
              endYear: nextEnd,
            })
          }}
          className="bg-transparent text-ink-primary font-semibold focus:outline-none cursor-pointer"
        >
          {years.map((yr) => (
            <option key={yr} value={yr}>
              {yr}
            </option>
          ))}
        </select>
      </div>

      {/* Active Quick-Remove Ticker Tags */}
      {filters.selectedTickers.map((ticker) => (
        <span
          key={ticker}
          className="inline-flex items-center gap-1 px-1.5 py-0.5 badge-amber text-[11px] font-bold"
        >
          {ticker}
          <button
            id={`btn-remove-ticker-${ticker}`}
            type="button"
            onClick={() => toggleTicker(ticker)}
            className="hover:opacity-75 cursor-pointer"
            aria-label={`Remove ${ticker} filter`}
          >
            <X className="w-3 h-3" aria-hidden="true" />
          </button>
        </span>
      ))}

      {hasCustomScope && (
        <button
          id="btn-reset-scope-filters"
          type="button"
          onClick={() =>
            onChangeFilters({
              selectedTickers: [],
              gicsSector: 'ALL',
              sectionId: 'ALL',
              startYear: 2024,
              endYear: 2026,
              topK: 8,
            })
          }
          className="text-[10px] uppercase tracking-wider text-bb-amber hover:underline ml-auto cursor-pointer"
        >
          RESET
        </button>
      )}
    </nav>
  )
}
