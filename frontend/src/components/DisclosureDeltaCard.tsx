import { Fragment, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import type {
  DisclosureDelta,
  DisclosureDeltaResult,
  GroundedCitation,
} from '../types/edgar'
import type { ParsedInlineCitationToken } from '../utils/formatters'
import { citationToken } from '../utils/citationTokens'
import { isRenderableDisclosureDeltas } from '../utils/deterministicEvidence'

interface DisclosureDeltaCardProps {
  turnId: string
  result: DisclosureDeltaResult
  citations: GroundedCitation[]
  onSelectCitation: (token: ParsedInlineCitationToken) => void
}

const CHANGE_STYLES: Record<DisclosureDelta['change_type'], { label: string; cls: string }> = {
  added: { label: '+ ADDED', cls: 'badge-positive' },
  removed: { label: '− REMOVED', cls: 'badge-negative' },
  modified: { label: '~ MODIFIED', cls: 'badge-amber' },
  unchanged: { label: '= SAME', cls: 'text-ink-muted border border-hairline' },
}

/** Year-over-year disclosure deltas: counts per year pair, expandable to cited passages. */
export function DisclosureDeltaCard({
  turnId,
  result,
  citations,
  onSelectCitation,
}: DisclosureDeltaCardProps) {
  const [expanded, setExpanded] = useState<string | null>(null)
  if (!isRenderableDisclosureDeltas(result)) {
    return null
  }
  const byId = new Map((citations ?? []).map((c) => [c.chunk_id, c]))
  const summary = result.summary

  const citeButton = (chunkId: string | null, label: string) => {
    const cit = chunkId ? byId.get(chunkId) : undefined
    if (!cit) return null
    return (
      <button
        type="button"
        onClick={() => onSelectCitation(citationToken(cit))}
        className="text-bb-cyan hover:underline text-[10px] font-semibold uppercase cursor-pointer"
        title={`${cit.ticker} FY${cit.fiscal_year} · ${cit.section_title}`}
      >
        [{label}]
      </button>
    )
  }

  return (
    <section
      aria-label="Year-over-Year Disclosure Deltas"
      className="mb-3 border border-hairline surface-panel overflow-hidden"
    >
      <div className="px-3 py-1.5 surface-header border-b border-hairline flex items-center justify-between gap-2 tabular-nums-mono">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-bb-amber">
          Disclosure Delta Engine
        </span>
        {summary && (
          <span className="text-[10px] text-ink-muted uppercase">
            +{summary.added} · ~{summary.modified} · −{summary.removed} ·{' '}
            {summary.companies_analyzed} ISSUER(S)
          </span>
        )}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-[11px] tabular-nums-mono">
          <thead>
            <tr className="surface-subtle border-b border-hairline text-[10px] text-ink-muted uppercase">
              <th className="py-1.5 px-2.5 font-medium">Equity</th>
              <th className="py-1.5 px-2.5 font-medium">Theme Mentions</th>
              <th className="py-1.5 px-2.5 font-medium">Year Pair</th>
              <th className="py-1.5 px-2.5 font-medium text-right">Added</th>
              <th className="py-1.5 px-2.5 font-medium text-right">Modified</th>
              <th className="py-1.5 px-2.5 font-medium text-right">Removed</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-hairline)]">
            {result.companies.map((comp) => {
              const mentions = Object.entries(comp.theme_mentions || {})
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([fy, n]) => `FY${fy.slice(-2)}:${n}`)
                .join(' → ')
              const yearPairs = comp.year_pairs ?? []
              const missingYears = comp.missing_fiscal_years ?? []
              const pairs = yearPairs.length > 0 ? yearPairs : [null]
              return pairs.map((pair, idx) => {
                const key = pair
                  ? `${comp.ticker}-${pair.prior_fiscal_year}-${pair.current_fiscal_year}`
                  : `${comp.ticker}-none`
                const isOpen = expanded === key
                const changes = pair
                  ? (pair.deltas ?? []).filter((d) => d.change_type !== 'unchanged')
                  : []
                return (
                  <Fragment key={key}>
                    <tr className="hover:bg-[var(--bg-subtle)] transition-colors">
                      <td className="py-1.5 px-2.5 whitespace-nowrap">
                        {idx === 0 && (
                          <span title={comp.company_name}>
                            <span className="font-bold text-bb-yellow">{comp.ticker}</span>
                            {missingYears.length > 0 && (
                              <span
                                className="ml-1.5 text-[9.5px] text-ink-muted"
                                title="No 10-K in corpus for these fiscal years"
                              >
                                (no FY
                                {missingYears
                                  .map((y) => String(y).slice(-2))
                                  .join('/')}
                                )
                              </span>
                            )}
                          </span>
                        )}
                      </td>
                      <td className="py-1.5 px-2.5 text-ink-secondary text-[10px] whitespace-nowrap">
                        {idx === 0 ? mentions || '—' : ''}
                      </td>
                      <td className="py-1.5 px-2.5 whitespace-nowrap">
                        {pair ? (
                          <button
                            id={`btn-expand-deltas-${turnId}-${key}`}
                            type="button"
                            onClick={() => setExpanded(isOpen ? null : key)}
                            className="inline-flex items-center gap-1 cursor-pointer text-ink-primary"
                            disabled={changes.length === 0}
                          >
                            {changes.length > 0 &&
                              (isOpen ? (
                                <ChevronDown className="w-3 h-3 text-bb-amber" />
                              ) : (
                                <ChevronRight className="w-3 h-3 text-ink-muted" />
                              ))}
                            FY{String(pair.prior_fiscal_year).slice(-2)}→FY
                            {String(pair.current_fiscal_year).slice(-2)}
                          </button>
                        ) : (
                          <span className="text-ink-muted">&lt; 2 filings</span>
                        )}
                      </td>
                      <td className="py-1.5 px-2.5 text-right text-bb-pos">
                        {pair?.counts?.added ?? '—'}
                      </td>
                      <td className="py-1.5 px-2.5 text-right text-bb-amber">
                        {pair?.counts?.modified ?? '—'}
                      </td>
                      <td className="py-1.5 px-2.5 text-right text-bb-neg">
                        {pair?.counts?.removed ?? '—'}
                      </td>
                    </tr>
                    {isOpen &&
                      changes.map((d) => {
                        const style = CHANGE_STYLES[d.change_type] ?? CHANGE_STYLES.modified
                        const phrases = [
                          ...(d.added_phrases ?? []).slice(0, 2).map((p) => `+ “${p}”`),
                          ...(d.removed_phrases ?? []).slice(0, 2).map((p) => `− “${p}”`),
                        ]
                        return (
                          <tr
                            key={d.delta_id}
                            className="surface-filing-sheet text-[10.5px] text-ink-secondary"
                          >
                            <td className="py-1 px-2.5 pl-7 align-top" colSpan={2}>
                              <span className={`px-1 text-[9.5px] font-bold ${style.cls}`}>
                                {style.label}
                              </span>
                              <div className="mt-1 font-sans text-ink-primary line-clamp-3">
                                {d.current_excerpt ?? d.prior_excerpt}
                              </div>
                              {phrases.length > 0 && (
                                <div className="mt-0.5 text-[10px] text-ink-muted line-clamp-2">
                                  {phrases.join('  ')}
                                </div>
                              )}
                            </td>
                            <td className="py-1 px-2.5 align-top" colSpan={4}>
                              <div className="flex flex-wrap gap-1.5 justify-end">
                                {citeButton(d.prior_chunk_id, `FY${d.prior_fiscal_year}`)}
                                {citeButton(d.current_chunk_id, `FY${d.current_fiscal_year}`)}
                              </div>
                            </td>
                          </tr>
                        )
                      })}
                  </Fragment>
                )
              })
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
