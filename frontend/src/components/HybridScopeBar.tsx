import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown, Filter, Search, X } from 'lucide-react'
import type {
  CompanySummary,
  SearchScopeFilters,
  SectionFilterId,
} from '../types/edgar'

interface HybridScopeBarProps {
  companies: CompanySummary[]
  sectors: string[]
  minYear: number
  maxYear: number
  filters: SearchScopeFilters
  onChangeFilters: (next: SearchScopeFilters) => void
}

const SECTION_OPTIONS: Array<{ id: SectionFilterId; label: string }> = [
  { id: 'ALL', label: 'All 10-K Sections (Hybrid)' },
  { id: 'item_7', label: 'Item 7 · MD&A' },
  { id: 'item_1a', label: 'Item 1A · Risk Factors' },
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
      className="surface-panel border-b border-hairline px-4 sm:px-6 py-2 flex flex-wrap items-center gap-2 z-10 shrink-0"
    >
      <div className="flex items-center gap-1.5 text-xs font-medium text-ink-muted mr-1">
        <Filter className="w-3.5 h-3.5" aria-hidden="true" />
        <span>Scope</span>
      </div>

      {/* Multi-Issuer Picker */}
      <div className="relative" ref={dropdownRef}>
        <button
          id="filter-issuer-trigger"
          type="button"
          onClick={() => setIsTickerMenuOpen((prev) => !prev)}
          aria-expanded={isTickerMenuOpen}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium surface-subtle border border-hairline text-ink-primary hover:border-strong cursor-pointer transition-colors"
        >
          <span>
            {filters.selectedTickers.length === 0
              ? 'Issuers: Auto / All S&P'
              : `Issuers: ${filters.selectedTickers.join(', ')}`}
          </span>
          <ChevronDown className="w-3.5 h-3.5 text-ink-muted" aria-hidden="true" />
        </button>

        {isTickerMenuOpen && (
          <div
            id="filter-issuer-popover"
            className="absolute left-0 mt-1.5 w-80 rounded-lg surface-elevated border border-hairline p-2 z-30"
          >
            <div className="relative mb-2">
              <Search
                className="w-3.5 h-3.5 text-ink-muted absolute left-2.5 top-1/2 -translate-y-1/2"
                aria-hidden="true"
              />
              <input
                id="input-issuer-search"
                type="text"
                value={tickerSearch}
                onChange={(e) => setTickerSearch(e.target.value)}
                placeholder="Filter ticker, company, or sector..."
                className="w-full pl-8 pr-2.5 py-1.5 text-xs rounded-md surface-subtle border border-hairline text-ink-primary focus:outline-none focus:border-strong"
              />
            </div>

            <div className="flex items-center justify-between px-1 pb-1.5 text-[11px] text-ink-muted border-b border-hairline mb-1">
              <span>Select 1+ for focused or peer comparison</span>
              {filters.selectedTickers.length > 0 && (
                <button
                  id="btn-clear-selected-issuers"
                  type="button"
                  onClick={() =>
                    onChangeFilters({ ...filters, selectedTickers: [] })
                  }
                  className="underline hover:text-ink-primary cursor-pointer"
                >
                  Clear ({filters.selectedTickers.length})
                </button>
              )}
            </div>

            <div className="max-h-60 overflow-y-auto space-y-0.5 pr-0.5">
              {filteredCompanies.map((company) => {
                const selected = filters.selectedTickers.includes(company.ticker)
                return (
                  <button
                    key={company.ticker}
                    id={`issuer-option-${company.ticker}`}
                    type="button"
                    onClick={() => toggleTicker(company.ticker)}
                    className={`w-full flex items-center justify-between px-2 py-1.5 rounded text-left text-xs cursor-pointer transition-colors ${
                      selected ? 'badge-accent font-medium' : 'hover:opacity-85 text-ink-primary'
                    }`}
                  >
                    <div className="min-w-0 pr-2">
                      <div className="flex items-center gap-1.5">
                        <span className="tabular-nums-mono font-semibold">
                          {company.ticker}
                        </span>
                        <span className="truncate text-ink-secondary">
                          {company.company_name}
                        </span>
                      </div>
                      <div className="text-[10px] text-ink-muted truncate">
                        {company.gics_sector} · FY{company.fiscal_years[0]}–FY
                        {company.latest_fiscal_year}
                      </div>
                    </div>
                    {selected && <Check className="w-3.5 h-3.5 shrink-0" />}
                  </button>
                )
              })}
              {filteredCompanies.length === 0 && (
                <p className="text-xs text-ink-muted py-3 text-center">
                  No matching curated issuers.
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* GICS Sector Filter */}
      <label className="sr-only" htmlFor="select-gics-sector">
        GICS Sector
      </label>
      <select
        id="select-gics-sector"
        value={filters.gicsSector}
        onChange={(e) =>
          onChangeFilters({ ...filters, gicsSector: e.target.value })
        }
        className="px-2.5 py-1 rounded-md text-xs font-medium surface-subtle border border-hairline text-ink-primary cursor-pointer focus:outline-none"
      >
        <option value="ALL">All GICS Sectors</option>
        {sectors.map((sec) => (
          <option key={sec} value={sec}>
            {sec}
          </option>
        ))}
      </select>

      {/* 10-K Section Filter */}
      <label className="sr-only" htmlFor="select-10k-section">
        Form 10-K Section
      </label>
      <select
        id="select-10k-section"
        value={filters.sectionId}
        onChange={(e) =>
          onChangeFilters({
            ...filters,
            sectionId: e.target.value as SectionFilterId,
          })
        }
        className="px-2.5 py-1 rounded-md text-xs font-medium surface-subtle border border-hairline text-ink-primary cursor-pointer focus:outline-none"
      >
        {SECTION_OPTIONS.map((opt) => (
          <option key={opt.id} value={opt.id}>
            {opt.label}
          </option>
        ))}
      </select>

      {/* Fiscal Period Range */}
      <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md surface-subtle border border-hairline text-xs tabular-nums-mono">
        <label htmlFor="select-start-year" className="text-ink-muted text-[11px]">
          FY
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
          className="bg-transparent text-ink-primary font-medium focus:outline-none cursor-pointer"
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
          className="bg-transparent text-ink-primary font-medium focus:outline-none cursor-pointer"
        >
          {years.map((yr) => (
            <option key={yr} value={yr}>
              {yr}
            </option>
          ))}
        </select>
      </div>

      {/* Active Quick-Remove Ticker Pills */}
      {filters.selectedTickers.map((ticker) => (
        <span
          key={ticker}
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded badge-accent text-[11px] tabular-nums-mono font-semibold"
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
          className="text-[11px] text-ink-muted hover:text-ink-primary underline ml-auto cursor-pointer"
        >
          Reset Scope
        </button>
      )}
    </nav>
  )
}
