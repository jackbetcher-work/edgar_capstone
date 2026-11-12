import { Fragment, useState } from 'react'
import { Calculator, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react'
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

function deltaToneClass(val: number | null | undefined): string {
  if (val === null || val === undefined || val === 0) return 'text-ink-secondary'
  return val > 0 ? 'badge-positive' : 'badge-negative'
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
      className="mb-4 rounded-lg surface-panel border border-hairline overflow-hidden"
    >
      <div className="px-3.5 py-2 surface-subtle border-b border-hairline flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Calculator className="w-3.5 h-3.5 text-ink-secondary" aria-hidden="true" />
          <span className="text-xs font-semibold text-ink-primary">
            Deterministic Variance Calculation Engine
          </span>
        </div>
        <span className="text-[11px] text-ink-muted tabular-nums-mono">
          100% XBRL Verified · USD
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-hairline text-[11px] text-ink-muted uppercase tracking-wider">
              <th className="py-2 px-3 font-medium">Issuer / Span</th>
              <th className="py-2 px-3 font-medium text-right">Revenue Δ</th>
              <th className="py-2 px-3 font-medium text-right">Rev CAGR</th>
              <th className="py-2 px-3 font-medium text-right">Op. Income Δ</th>
              <th className="py-2 px-3 font-medium text-right">Op. Margin</th>
              <th className="py-2 px-3 font-medium text-right">Net Income Δ</th>
              <th className="py-2 px-3 font-medium text-right">10-K</th>
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
                  <tr className="hover:bg-[var(--bg-subtle)]/50 transition-colors">
                    <td className="py-2.5 px-3">
                      <button
                        id={`btn-expand-annual-${turnId}-${rep.ticker}`}
                        type="button"
                        onClick={() =>
                          setExpandedTicker(isExpanded ? null : rep.ticker)
                        }
                        className="inline-flex items-center gap-1.5 text-left cursor-pointer group"
                        title="Toggle annual step-by-step YoY breakdown"
                      >
                        {isExpanded ? (
                          <ChevronDown className="w-3.5 h-3.5 text-ink-muted" />
                        ) : (
                          <ChevronRight className="w-3.5 h-3.5 text-ink-muted" />
                        )}
                        <div>
                          <div className="font-semibold text-ink-primary group-hover:underline">
                            <span className="tabular-nums-mono">{rep.ticker}</span>{' '}
                            <span className="font-normal text-ink-secondary hidden sm:inline">
                              · {rep.company_name}
                            </span>
                          </div>
                          <div className="text-[10px] text-ink-muted tabular-nums-mono">
                            FY{rep.start_year} → FY{rep.end_year} · {rep.gics_sector}
                          </div>
                        </div>
                      </button>
                    </td>

                    {/* Revenue Variance */}
                    <td className="py-2.5 px-3 text-right tabular-nums-mono">
                      <div className="font-medium text-ink-primary">
                        {formatUsdCompact(rev.start_value)} →{' '}
                        {formatUsdCompact(rev.end_value)}
                      </div>
                      <span
                        className={`inline-block mt-0.5 px-1.5 py-0.2 rounded text-[11px] font-medium ${deltaToneClass(
                          rev.percentage_delta
                        )}`}
                      >
                        {formatSignedPercent(rev.percentage_delta)} (
                        {formatSignedDeltaUsd(rev.absolute_delta)})
                      </span>
                    </td>

                    {/* CAGR */}
                    <td className="py-2.5 px-3 text-right tabular-nums-mono text-ink-primary font-medium">
                      {formatSignedPercent(rev.cagr_pct)}
                    </td>

                    {/* Operating Income Variance */}
                    <td className="py-2.5 px-3 text-right tabular-nums-mono">
                      <div className="text-ink-primary font-medium">
                        {formatUsdCompact(op.end_value)}
                      </div>
                      <span
                        className={`inline-block mt-0.5 px-1.5 py-0.2 rounded text-[11px] font-medium ${deltaToneClass(
                          op.percentage_delta
                        )}`}
                      >
                        {formatSignedPercent(op.percentage_delta)}
                      </span>
                    </td>

                    {/* Operating Margin bps */}
                    <td className="py-2.5 px-3 text-right tabular-nums-mono">
                      <span
                        className={`inline-block px-1.5 py-0.2 rounded text-[11px] font-medium ${deltaToneClass(
                          rep.operating_margin_bps_delta
                        )}`}
                      >
                        {formatSignedBps(rep.operating_margin_bps_delta)}
                      </span>
                    </td>

                    {/* Net Income Variance */}
                    <td className="py-2.5 px-3 text-right tabular-nums-mono">
                      <div className="text-ink-primary font-medium">
                        {formatUsdCompact(ni.end_value)}
                      </div>
                      <span
                        className={`inline-block mt-0.5 px-1.5 py-0.2 rounded text-[11px] font-medium ${deltaToneClass(
                          ni.percentage_delta
                        )}`}
                      >
                        {formatSignedPercent(ni.percentage_delta)}
                      </span>
                    </td>

                    {/* Inspect 10-K Action */}
                    <td className="py-2.5 px-3 text-right">
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
                            className="px-2 py-0.5 rounded badge-accent text-[11px] font-medium cursor-pointer hover:opacity-85"
                            title={`Open ${rep.ticker} FY${rep.end_year} Item 7 MD&A in Full Filing Reader`}
                          >
                            MD&amp;A
                          </button>
                        )}
                        {latestUrl && (
                          <a
                            id={`link-sec-edgar-${turnId}-${rep.ticker}`}
                            href={latestUrl}
                            target="_blank"
                            rel="noreferrer"
                            className="p-1 text-ink-muted hover:text-ink-primary"
                            title={`Open official ${rep.ticker} FY${rep.end_year} SEC EDGAR filing`}
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </a>
                        )}
                      </div>
                    </td>
                  </tr>

                  {/* Expandable Annual Step Series */}
                  {isExpanded && rep.annual_series.length > 0 && (
                    <tr className="surface-subtle">
                      <td colSpan={7} className="px-4 py-2.5">
                        <div className="text-[11px] font-medium text-ink-secondary mb-1.5">
                          Annual YoY Trajectory — {rep.company_name} ({rep.ticker})
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                          {rep.annual_series.map((step) => (
                            <div
                              key={`${rep.ticker}-${step.fiscal_year}`}
                              className="rounded surface-panel border border-hairline p-2 tabular-nums-mono text-[11px]"
                            >
                              <div className="flex items-center justify-between font-semibold text-ink-primary mb-1">
                                <span>FY{step.fiscal_year}</span>
                                {step.revenue_yoy_pct !== null && (
                                  <span
                                    className={`px-1.5 py-0.2 rounded text-[10px] ${deltaToneClass(
                                      step.revenue_yoy_pct
                                    )}`}
                                  >
                                    Rev {formatSignedPercent(step.revenue_yoy_pct)} YoY
                                  </span>
                                )}
                              </div>
                              <div className="flex justify-between text-ink-secondary">
                                <span>Revenue:</span>
                                <span className="text-ink-primary font-medium">
                                  {formatUsdCompact(step.revenue)}
                                </span>
                              </div>
                              <div className="flex justify-between text-ink-secondary">
                                <span>Op. Income:</span>
                                <span className="text-ink-primary">
                                  {formatUsdCompact(step.operating_income)}
                                </span>
                              </div>
                              <div className="flex justify-between text-ink-secondary">
                                <span>Net Income:</span>
                                <span className="text-ink-primary">
                                  {formatUsdCompact(step.net_income)}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
