import { Fragment, useState } from 'react'
import { Calendar, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react'
import type {
  CompanyVarianceReport,
  MetricVariance,
  XbrlFactSelection,
} from '../types/edgar'
import {
  formatPeriodEndShort,
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
  onInspectXbrlFact?: (fact: XbrlFactSelection) => void
}

function deltaTextClass(val: number | null | undefined): string {
  if (val === null || val === undefined || val === 0) return 'text-ink-secondary'
  return val > 0 ? 'text-bb-pos' : 'text-bb-neg'
}

function extractMonth(isoDate: string | null | undefined): string | null {
  if (!isoDate) return null
  const m = /^\d{4}-(\d{2})-\d{2}/.exec(isoDate.trim())
  return m ? m[1] : null
}

function buildVarianceXbrlSelection(
  rep: CompanyVarianceReport,
  mv: MetricVariance,
  metricLabel: string
): XbrlFactSelection {
  const startKey = String(mv.start_year || rep.start_year)
  const endKey = String(mv.end_year || rep.end_year)
  return {
    title: `${rep.ticker} ${metricLabel} (FY${mv.start_year}→FY${mv.end_year})`,
    ticker: rep.ticker,
    companyName: rep.company_name,
    fiscalYear: mv.end_year || rep.end_year,
    priorFiscalYear: mv.start_year || rep.start_year,
    metricLabel,
    valueFormatted: `${formatUsdCompact(mv.start_value)} → ${formatUsdCompact(mv.end_value)}`,
    deltaFormatted:
      mv.percentage_delta !== null
        ? `${formatSignedPercent(mv.percentage_delta)} (${formatSignedDeltaUsd(mv.absolute_delta)})`
        : undefined,
    periodOfReport:
      mv.end_period_of_report ?? rep.period_end_dates?.[endKey] ?? null,
    priorPeriodOfReport:
      mv.start_period_of_report ?? rep.period_end_dates?.[startKey] ?? null,
    filingDate: rep.filing_dates?.[endKey] ?? null,
    accessionNumber:
      mv.end_accession_number ?? rep.accession_numbers?.[endKey] ?? null,
    priorAccessionNumber:
      mv.start_accession_number ?? rep.accession_numbers?.[startKey] ?? null,
    xbrlConcept:
      mv.end_xbrl_concept ??
      rep.metric_provenance?.[endKey]?.[mv.metric_name] ??
      null,
    priorXbrlConcept:
      mv.start_xbrl_concept ??
      rep.metric_provenance?.[startKey]?.[mv.metric_name] ??
      null,
    secFilingUrl:
      mv.end_sec_filing_url ?? rep.source_urls?.[endKey] ?? null,
    priorSecFilingUrl:
      mv.start_sec_filing_url ?? rep.source_urls?.[startKey] ?? null,
    derivedFrom: mv.derived_from,
  }
}

export function VarianceTableCard({
  turnId,
  reports,
  onInspectFilingSection,
  onInspectXbrlFact,
}: VarianceTableCardProps) {
  const [expandedTicker, setExpandedTicker] = useState<string | null>(
    reports.length === 1 ? reports[0].ticker : null
  )

  if (!reports || reports.length === 0) return null

  const fiscalMonths = new Set<string>()
  reports.forEach((rep) => {
    const endKey = String(rep.end_year)
    const period =
      rep.revenue_variance.end_period_of_report ??
      rep.period_end_dates?.[endKey] ??
      rep.annual_series[rep.annual_series.length - 1]?.period_of_report
    const month = extractMonth(period)
    if (month) fiscalMonths.add(month)
  })
  const hasFiscalMisalignment = reports.length > 1 && fiscalMonths.size > 1

  return (
    <section
      aria-label="Deterministic Financial Variance Engine Output"
      className="mb-3 border border-hairline surface-panel overflow-hidden"
    >
      <div className="px-3 py-1.5 surface-header border-b border-hairline flex flex-wrap items-center justify-between gap-2 tabular-nums-mono">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-bb-amber">
          Deterministic Variance Calculation Engine
        </span>
        <div className="flex items-center gap-2">
          {hasFiscalMisalignment && (
            <span
              className="inline-flex items-center gap-1 px-1.5 py-0.2 text-[9.5px] font-semibold uppercase badge-amber"
              title="Compared peers have different fiscal year-end months; verify period-end dates below."
            >
              <Calendar className="w-2.5 h-2.5" />
              DIFFERING FY-END DATES
            </span>
          )}
          <span className="text-[10px] text-ink-muted uppercase">
            XBRL AUDITED · USD
          </span>
        </div>
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
              <th className="py-1.5 px-2.5 font-medium text-right">R&amp;D Δ</th>
              <th className="py-1.5 px-2.5 font-medium text-right">10-K</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-hairline)]">
            {reports.map((rep) => {
              const isExpanded = expandedTicker === rep.ticker
              const rev = rep.revenue_variance
              const op = rep.operating_income_variance
              const ni = rep.net_income_variance
              const rd = rep.research_and_development_variance
              const hasRd =
                rd != null && rd.start_value !== null && rd.end_value !== null
              const latestUrl =
                rep.source_urls[String(rep.end_year)] ||
                rep.annual_series[rep.annual_series.length - 1]?.sec_filing_url
              const startPeriod =
                rev.start_period_of_report ??
                rep.period_end_dates?.[String(rep.start_year)] ??
                null
              const endPeriod =
                rev.end_period_of_report ??
                rep.period_end_dates?.[String(rep.end_year)] ??
                null
              const periodSpanLabel =
                startPeriod || endPeriod
                  ? `${formatPeriodEndShort(startPeriod) || `FY${rep.start_year}`} → ${formatPeriodEndShort(endPeriod) || `FY${rep.end_year}`}`
                  : null
              const hasXbrlMeta = Boolean(
                rev.end_accession_number ||
                  rev.end_xbrl_concept ||
                  rep.accession_numbers?.[String(rep.end_year)] ||
                  rep.metric_provenance?.[String(rep.end_year)]
              )

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
                          <div className="text-[10px] text-ink-secondary truncate max-w-[165px] font-sans">
                            {rep.company_name}
                          </div>
                          {periodSpanLabel && (
                            <div
                              className="text-[9.5px] text-ink-muted"
                              title={`Fiscal period end dates: ${startPeriod ?? '—'} to ${endPeriod ?? '—'}`}
                            >
                              {periodSpanLabel}
                            </div>
                          )}
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

                    <td
                      className="py-2 px-2.5 text-right whitespace-nowrap"
                      title={
                        hasRd
                          ? `R&D intensity ${formatSignedBps(rep.rd_intensity_bps_delta ?? null)}`
                          : 'No R&D expense reported in XBRL'
                      }
                    >
                      {hasRd ? (
                        <>
                          <div className="text-ink-primary font-medium">
                            {formatUsdCompact(rd.end_value)}
                          </div>
                          <div
                            className={`text-[10px] font-medium ${deltaTextClass(
                              rd.percentage_delta
                            )}`}
                          >
                            {formatSignedPercent(rd.percentage_delta)}
                          </div>
                        </>
                      ) : (
                        <span className="text-ink-muted">—</span>
                      )}
                    </td>

                    <td className="py-2 px-2.5 text-right whitespace-nowrap">
                      <div className="inline-flex items-center gap-1.5 justify-end">
                        {onInspectXbrlFact && hasXbrlMeta && (
                          <button
                            id={`btn-inspect-xbrl-${turnId}-${rep.ticker}`}
                            type="button"
                            onClick={() =>
                              onInspectXbrlFact(
                                buildVarianceXbrlSelection(rep, rev, 'Revenue')
                              )
                            }
                            className="text-bb-amber hover:underline text-[10px] font-semibold uppercase cursor-pointer"
                            title={`Inspect ${rep.ticker} XBRL tag provenance & accession numbers`}
                          >
                            [XBRL]
                          </button>
                        )}
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
                    rep.annual_series.map((step) => {
                      const stepPeriod =
                        step.period_of_report ??
                        rep.period_end_dates?.[String(step.fiscal_year)] ??
                        null
                      const stepAcc =
                        step.accession_number ??
                        rep.accession_numbers?.[String(step.fiscal_year)] ??
                        null
                      const stepRevConcept =
                        step.metric_provenance?.revenue ??
                        rep.metric_provenance?.[String(step.fiscal_year)]
                          ?.revenue ??
                        null
                      return (
                        <tr
                          key={`${rep.ticker}-annual-${step.fiscal_year}`}
                          className="surface-filing-sheet text-[10.5px] text-ink-secondary"
                        >
                          <td className="py-1 px-2.5 pl-7 text-ink-muted whitespace-nowrap">
                            <span>└ FY{step.fiscal_year} ANNUAL</span>
                            {stepPeriod && (
                              <span className="ml-1.5 text-[9.5px] text-ink-muted">
                                ({stepPeriod})
                              </span>
                            )}
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
                          <td
                            className={`py-1 px-2.5 text-right ${deltaTextClass(
                              step.research_and_development_yoy_pct ?? null
                            )}`}
                          >
                            {step.research_and_development != null
                              ? `R&D ${formatUsdCompact(step.research_and_development)}`
                              : '—'}
                          </td>
                          <td className="py-1 px-2.5 text-right whitespace-nowrap">
                            <div className="inline-flex items-center gap-1.5 justify-end">
                              {onInspectXbrlFact && (stepAcc || stepRevConcept) && (
                                <button
                                  type="button"
                                  onClick={() =>
                                    onInspectXbrlFact({
                                      title: `${rep.ticker} FY${step.fiscal_year} Annual Filing Facts`,
                                      ticker: rep.ticker,
                                      companyName: rep.company_name,
                                      fiscalYear: step.fiscal_year,
                                      priorFiscalYear: step.prev_fiscal_year,
                                      metricLabel: 'Revenue',
                                      valueFormatted: formatUsdCompact(step.revenue),
                                      deltaFormatted:
                                        step.revenue_yoy_pct !== null
                                          ? `${formatSignedPercent(step.revenue_yoy_pct)} YoY`
                                          : undefined,
                                      periodOfReport: stepPeriod,
                                      priorPeriodOfReport:
                                        step.prior_period_of_report ?? null,
                                      filingDate:
                                        step.filing_date ??
                                        rep.filing_dates?.[
                                          String(step.fiscal_year)
                                        ] ??
                                        null,
                                      accessionNumber: stepAcc,
                                      priorAccessionNumber:
                                        step.prior_accession_number ?? null,
                                      xbrlConcept: stepRevConcept,
                                      secFilingUrl: step.sec_filing_url,
                                      priorSecFilingUrl:
                                        step.prior_sec_filing_url ?? null,
                                    })
                                  }
                                  className="text-[9.5px] text-bb-amber hover:underline cursor-pointer"
                                  title={`XBRL acc:${stepAcc ?? 'n/a'} · ${stepRevConcept ?? 'XBRL'}`}
                                >
                                  [XBRL]
                                </button>
                              )}
                              {step.sec_filing_url && (
                                <a
                                  href={step.sec_filing_url}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="text-ink-muted hover:text-ink-primary inline-flex justify-end"
                                  title={`SEC EDGAR FY${step.fiscal_year} 10-K${stepAcc ? ` · acc:${stepAcc}` : ''}`}
                                >
                                  <ExternalLink className="w-2.5 h-2.5" />
                                </a>
                              )}
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
