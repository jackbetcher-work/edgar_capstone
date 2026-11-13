import { Fragment, useState } from 'react'
import { ChevronDown, ChevronRight, ExternalLink } from 'lucide-react'
import type { CompanyVarianceReport } from '../types/edgar'
import {
  formatSignedBps,
  formatSignedDeltaUsd,
  formatSignedPercent,
  formatUsdCompact,
} from '../utils/formatters'

interface VarianceTableCardProps {
  turnId: string
  reports: CompanyVarianceReport[]
  onInspectFilingSection?: (
    ticker: string,
    fiscalYear: number,
    sectionId: 'item_7' | 'item_1a'
  ) => void
}

function deltaTextClass(val: number | null | undefined): string {
  if (val === null || val === undefined || val === 0) return 'text-ink-secondary'
  return val > 0 ? 'text-bb-pos' : 'text-bb-neg'
}

export function VarianceTableCard({
  turnId,
  reports,
  onInspectFilingSection,
}: VarianceTableCardProps) {
  const [expandedTicker, setExpandedTicker] = useState<string | null>(
    reports.length === 1 ? reports[0].ticker : null
  )

  if (!reports || reports.length === 0) return null

  return (
    <section
      aria-label="Deterministic Financial Variance Engine Output"
      className="mb-3 border border-hairline surface-panel overflow-hidden"
    >
      <div className="px-3 py-1.5 surface-header border-b border-hairline flex items-center justify-between gap-2 tabular-nums-mono">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-bb-amber">
          Deterministic Variance Calculation Engine
        </span>
        <span className="text-[10px] text-ink-muted uppercase">
          XBRL AUDITED · USD
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-[11px] tabular-nums-mono">
          <thead>
            <tr className="surface-subtle border-b border-hairline text-[10px] text-ink-muted uppercase">
              <th className="py-1.5 px-2.5 font-medium">Equity / Span</th>
              <th className="py-1.5 px-2.5 font-medium text-right">
                Rev Start → End
              </th>
              <th className="py-1.5 px-2.5 font-medium text-right">Rev Δ</th>
              <th className="py-1.5 px-2.5 font-medium text-right">CAGR</th>
              <th className="py-1.5 px-2.5 font-medium text-right">Op Inc Δ</th>
              <th className="py-1.5 px-2.5 font-medium text-right">Op Mgn</th>
              <th className="py-1.5 px-2.5 font-medium text-right">Net Inc Δ</th>
              <th className="py-1.5 px-2.5 font-medium text-right">10-K</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-hairline)]">
            {reports.map((rep) => {
              const isExpanded = expandedTicker === rep.ticker
              const rev = rep.revenue_variance
              const op = rep.operating_income_variance
              const ni = rep.net_income_variance
              const latestUrl =
                rep.source_urls[String(rep.end_year)] ||
                rep.annual_series[rep.annual_series.length - 1]?.sec_filing_url

              return (
                <Fragment key={`${rep.ticker}-${rep.start_year}-${rep.end_year}`}>
                  <tr className="hover:bg-[var(--bg-subtle)] transition-colors">
                    <td className="py-2 px-2.5">
                      <button
                        id={`btn-expand-annual-${turnId}-${rep.ticker}`}
                        type="button"
                        onClick={() =>
                          setExpandedTicker(isExpanded ? null : rep.ticker)
                        }
                        className="inline-flex items-center gap-1.5 text-left cursor-pointer group"
                        title="Toggle annual YoY trajectory"
                      >
                        {isExpanded ? (
                          <ChevronDown className="w-3 h-3 text-bb-amber shrink-0" />
                        ) : (
                          <ChevronRight className="w-3 h-3 text-ink-muted shrink-0" />
                        )}
                        <div>
                          <div className="flex items-center gap-1.5">
                            <span className="font-bold text-bb-yellow group-hover:underline">
                              {rep.ticker}
                            </span>
                            <span className="text-ink-muted text-[10px]">
                              FY{String(rep.start_year).slice(-2)}→FY
                              {String(rep.end_year).slice(-2)}
                            </span>
                          </div>
                          <div className="text-[10px] text-ink-secondary truncate max-w-[150px] font-sans">
                            {rep.company_name}
                          </div>
                        </div>
                      </button>
                    </td>

                    <td className="py-2 px-2.5 text-right text-ink-primary whitespace-nowrap">
                      {formatUsdCompact(rev.start_value)} →{' '}
                      <span className="font-semibold">
                        {formatUsdCompact(rev.end_value)}
                      </span>
                    </td>

                    <td className="py-2 px-2.5 text-right whitespace-nowrap">
                      <div
                        className={`font-semibold ${deltaTextClass(
                          rev.percentage_delta
                        )}`}
                      >
                        {formatSignedPercent(rev.percentage_delta)}
                      </div>
                      <div className="text-[10px] text-ink-muted">
                        {formatSignedDeltaUsd(rev.absolute_delta)}
                      </div>
                    </td>

                    <td
                      className={`py-2 px-2.5 text-right font-medium whitespace-nowrap ${deltaTextClass(
                        rev.cagr_pct
                      )}`}
                    >
                      {formatSignedPercent(rev.cagr_pct)}
                    </td>

                    <td className="py-2 px-2.5 text-right whitespace-nowrap">
                      <div className="text-ink-primary font-medium">
                        {formatUsdCompact(op.end_value)}
                      </div>
                      <div
                        className={`text-[10px] font-medium ${deltaTextClass(
                          op.percentage_delta
                        )}`}
                      >
                        {formatSignedPercent(op.percentage_delta)}
                      </div>
                    </td>

                    <td
                      className={`py-2 px-2.5 text-right font-semibold whitespace-nowrap ${deltaTextClass(
                        rep.operating_margin_bps_delta
                      )}`}
                    >
                      {formatSignedBps(rep.operating_margin_bps_delta)}
                    </td>

                    <td className="py-2 px-2.5 text-right whitespace-nowrap">
                      <div className="text-ink-primary font-medium">
                        {formatUsdCompact(ni.end_value)}
                      </div>
                      <div
                        className={`text-[10px] font-medium ${deltaTextClass(
                          ni.percentage_delta
                        )}`}
                      >
                        {formatSignedPercent(ni.percentage_delta)}
                      </div>
                    </td>

                    <td className="py-2 px-2.5 text-right whitespace-nowrap">
                      <div className="inline-flex items-center gap-1.5 justify-end">
                        {onInspectFilingSection && (
                          <button
                            id={`btn-open-mdna-${turnId}-${rep.ticker}`}
                            type="button"
                            onClick={() =>
                              onInspectFilingSection(
                                rep.ticker,
                                rep.end_year,
                                'item_7'
                              )
                            }
                            className="text-bb-cyan hover:underline text-[10px] font-semibold uppercase cursor-pointer"
                            title={`Open ${rep.ticker} FY${rep.end_year} Item 7 MD&A`}
                          >
                            [MD&amp;A]
                          </button>
                        )}
                        {latestUrl && (
                          <a
                            id={`link-sec-edgar-${turnId}-${rep.ticker}`}
                            href={latestUrl}
                            target="_blank"
                            rel="noreferrer"
                            className="text-ink-muted hover:text-ink-primary"
                            title={`Open official ${rep.ticker} FY${rep.end_year} SEC EDGAR filing`}
                          >
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>
                    </td>
                  </tr>

                  {/* Inline Annual YoY Sub-Rows (Flat Terminal Sub-Grid, Zero Nested Cards) */}
                  {isExpanded &&
                    rep.annual_series.map((step) => (
                      <tr
                        key={`${rep.ticker}-annual-${step.fiscal_year}`}
                        className="surface-filing-sheet text-[10.5px] text-ink-secondary"
                      >
                        <td className="py-1 px-2.5 pl-7 text-ink-muted">
                          └ FY{step.fiscal_year} ANNUAL
                        </td>
                        <td className="py-1 px-2.5 text-right text-ink-primary">
                          REV {formatUsdCompact(step.revenue)}
                        </td>
                        <td
                          className={`py-1 px-2.5 text-right ${deltaTextClass(
                            step.revenue_yoy_pct
                          )}`}
                        >
                          {step.revenue_yoy_pct !== null
                            ? `${formatSignedPercent(step.revenue_yoy_pct)} YoY`
                            : '—'}
                        </td>
                        <td className="py-1 px-2.5 text-right text-ink-muted">
                          —
                        </td>
                        <td className="py-1 px-2.5 text-right text-ink-primary">
                          OP {formatUsdCompact(step.operating_income)}
                        </td>
                        <td
                          className={`py-1 px-2.5 text-right ${deltaTextClass(
                            step.operating_income_yoy_pct
                          )}`}
                        >
                          {step.operating_income_yoy_pct !== null
                            ? `${formatSignedPercent(step.operating_income_yoy_pct)} YoY`
                            : '—'}
                        </td>
                        <td className="py-1 px-2.5 text-right text-ink-primary">
                          NI {formatUsdCompact(step.net_income)}
                        </td>
                        <td className="py-1 px-2.5 text-right">
                          {step.sec_filing_url && (
                            <a
                              href={step.sec_filing_url}
                              target="_blank"
                              rel="noreferrer"
                              className="text-ink-muted hover:text-ink-primary inline-flex justify-end"
                              title={`SEC EDGAR FY${step.fiscal_year} 10-K`}
                            >
                              <ExternalLink className="w-2.5 h-2.5" />
                            </a>
                          )}
                        </td>
                      </tr>
                    ))}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
