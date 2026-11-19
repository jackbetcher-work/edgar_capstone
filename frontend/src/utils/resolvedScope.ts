import type { ResolvedQueryScope } from '../types/edgar'

export interface ScopeSummary {
  /** Companies or sector the agent searched, e.g. "AAPL, MSFT" or "Information Technology". */
  issuers: string
  /** Full ticker list / sector for the tooltip when `issuers` is abbreviated. */
  issuersFull: string
  /** Fiscal-year window, e.g. "FY2022–FY2024". */
  years: string
  /** 10-K sections searched for narrative evidence. */
  sections: string
  /** True when companies/years were carried over from earlier turns. */
  carriedOver: boolean
}

const MAX_TICKERS_SHOWN = 4

/**
 * Human-readable summary of the scope the backend resolved from the analyst's
 * question, so the analyst can confirm what was searched without scope filters.
 */
export function summarizeResolvedScope(
  scope: ResolvedQueryScope | null | undefined
): ScopeSummary | null {
  if (!scope) return null
  const tickers = (scope.tickers ?? []).filter(Boolean)
  const sector = scope.gics_sector && scope.gics_sector.toUpperCase() !== 'ALL' ? scope.gics_sector : null

  let issuers: string
  let issuersFull: string
  if (tickers.length > 0) {
    const shown = tickers.slice(0, MAX_TICKERS_SHOWN).join(', ')
    const extra = tickers.length - MAX_TICKERS_SHOWN
    issuers = extra > 0 ? `${shown} +${extra}` : shown
    issuersFull = tickers.join(', ')
  } else if (sector) {
    issuers = sector
    issuersFull = `${sector} sector`
  } else {
    issuers = 'All companies'
    issuersFull = 'All companies in the corpus'
  }

  const start = Number(scope.start_year)
  const end = Number(scope.end_year)
  const years =
    Number.isFinite(start) && Number.isFinite(end) && start > 0
      ? start === end
        ? `FY${start}`
        : `FY${start}–FY${end}`
      : ''

  const sectionKey = (scope.section_id ?? 'ALL').toLowerCase()
  const sections =
    sectionKey === 'item_1a'
      ? 'Risk Factors'
      : sectionKey === 'item_7'
        ? 'MD&A'
        : 'MD&A + Risk Factors'

  return {
    issuers,
    issuersFull,
    years,
    sections,
    carriedOver: Boolean(scope.memory_inherited),
  }
}
