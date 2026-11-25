import { useState } from 'react'
import { ChevronDown, ChevronRight, Database, FileText } from 'lucide-react'
import type {
  CompanyVarianceReport,
  GroundedCitation,
  MetricVariance,
  StructuredAnalystReport,
  StructuredKeyMetric,
  XbrlFactSelection,
} from '../types/edgar'
import {
  formatSignedDeltaUsd,
  formatSignedPercent,
  formatUsdCompact,
  type ParsedInlineCitationToken,
} from '../utils/formatters'
import { buildXbrlFactFromVariance } from '../utils/xbrlFactSelection'
import { shortCitationTopic } from '../utils/xbrlLabels'

interface StructuredReportViewProps {
  turnId: string
  report: StructuredAnalystReport
  varianceReports: CompanyVarianceReport[]
  citations: GroundedCitation[]
  isChunkActive: (chunkId: string) => boolean
  onSelectCitation: (token: ParsedInlineCitationToken) => void
  onInspectXbrlFact?: (fact: XbrlFactSelection) => void
}

const METRIC_LABELS: Record<StructuredKeyMetric['metric'], string> = {
  revenue: 'Revenue',
  operating_income: 'Operating Income',
  net_income: 'Net Income',
  research_and_development: 'R&D Expense',
}

function resolveVarianceMetric(
  rep: CompanyVarianceReport,
  metric: StructuredKeyMetric['metric']
): MetricVariance | null {
  switch (metric) {
    case 'revenue':
      return rep.revenue_variance
    case 'operating_income':
      return rep.operating_income_variance
    case 'net_income':
      return rep.net_income_variance
    case 'research_and_development':
      return rep.research_and_development_variance ?? null
  }
}

function formatKeyMetricSummary(
  rep: CompanyVarianceReport,
  metric: StructuredKeyMetric['metric']
): { text: string; fact: XbrlFactSelection | null } | null {
  const label = METRIC_LABELS[metric]
  const mv = resolveVarianceMetric(rep, metric)
  if (mv && mv.direction !== 'unavailable') {
    const startVal = formatUsdCompact(mv.start_value)
    const endVal = formatUsdCompact(mv.end_value)
    const pct =
      mv.percentage_delta !== null ? formatSignedPercent(mv.percentage_delta) : null
    const absDelta =
      mv.absolute_delta !== null ? formatSignedDeltaUsd(mv.absolute_delta) : null
    const deltaPart = [pct, absDelta].filter(Boolean).join(', ')
    const text = `FY${mv.start_year} ${startVal} → FY${mv.end_year} ${endVal}${
      deltaPart ? ` (${deltaPart})` : ''
    }`
    return {
      text,
      fact: buildXbrlFactFromVariance(rep, mv, label),
    }
  }
  return null
}

function EvidenceQuoteToggle({
  itemKey,
  quote,
}: {
  itemKey: string
  quote: string
}) {
  const [open, setOpen] = useState(false)
  if (!quote.trim()) return null
  return (
    <div className="mt-0.5">
      <button
        id={`btn-evidence-${itemKey}`}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="inline-flex items-center gap-0.5 text-[10.5px] tabular-nums-mono uppercase tracking-wide text-ink-muted hover:text-ink-primary cursor-pointer"
      >
        {open ? (
          <ChevronDown className="w-3 h-3" aria-hidden="true" />
        ) : (
          <ChevronRight className="w-3 h-3" aria-hidden="true" />
        )}
        Supporting quote
      </button>
      {open && (
        <blockquote className="mt-1 px-2.5 py-1.5 text-[11.5px] leading-relaxed border-l-2 border-l-[var(--border-strong)] surface-subtle text-ink-secondary italic">
          &ldquo;{quote.trim()}&rdquo;
        </blockquote>
      )}
    </div>
  )
}

/**
 * Direct JSON renderer for verified `StructuredAnalystReport` responses.
 * Eliminates backend-Markdown → frontend-regex round-tripping for structured turns.
 */
export function StructuredReportView({
  turnId,
  report,
  varianceReports,
  citations,
  isChunkActive,
  onSelectCitation,
  onInspectXbrlFact,
}: StructuredReportViewProps) {
  const varianceByTicker = new Map(
    varianceReports.map((r) => [r.ticker.toUpperCase(), r])
  )
  const citationById = new Map(citations.map((c) => [c.chunk_id, c]))

  const renderedMetrics = (report.key_metrics ?? [])
    .map((km, idx) => {
      const rep = varianceByTicker.get(km.ticker.toUpperCase())
      if (!rep) return null
      const summary = formatKeyMetricSummary(rep, km.metric)
      if (!summary) return null
      return { km, rep, summary, idx }
    })
    .filter((entry): entry is NonNullable<typeof entry> => entry !== null)

  return (
    <div className="space-y-1">
      {report.headline && (
        <h3 className="tabular-nums-mono text-xs font-semibold uppercase tracking-wider text-bb-amber mt-0 pb-1 mb-1.5 border-b border-hairline">
          {report.headline}
        </h3>
      )}

      {renderedMetrics.length > 0 && (
        <div>
          <h4 className="tabular-nums-mono text-[11px] font-semibold uppercase tracking-wider text-ink-muted mt-3 mb-1">
            Key Metrics
          </h4>
          <ul className="space-y-1.5 my-2 pl-4 list-square marker:text-[var(--bb-amber)]">
            {renderedMetrics.map(({ km, rep, summary, idx }) => (
              <li
                key={`${turnId}-km-${km.ticker}-${km.metric}-${idx}`}
                className="text-[12.5px] leading-relaxed text-ink-secondary pl-0.5"
              >
                <strong className="font-semibold text-ink-primary">
                  {rep.ticker} {METRIC_LABELS[km.metric]}:
                </strong>{' '}
                <span className="tabular-nums-mono text-ink-primary">
                  {summary.text}
                </span>
                {summary.fact && onInspectXbrlFact && (
                  <>
                    {' '}
                    <button
                      id={`structured-xbrl-${turnId}-${idx}`}
                      type="button"
                      onClick={() => onInspectXbrlFact(summary.fact!)}
                      className="citation-chip"
                      title={`Inspect XBRL provenance for ${rep.ticker} ${METRIC_LABELS[km.metric]}`}
                    >
                      <Database className="w-2.5 h-2.5 shrink-0" aria-hidden="true" />
                      <span>
                        {rep.ticker} FY{String(rep.end_year).slice(-2)} ·{' '}
                        {METRIC_LABELS[km.metric]}
                      </span>
                    </button>
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {report.findings && report.findings.length > 0 && (
        <div>
          <h4 className="tabular-nums-mono text-[11px] font-semibold uppercase tracking-wider text-ink-muted mt-3 mb-1">
            10-K Findings
          </h4>
          <ul className="space-y-2 my-2 pl-4 list-square marker:text-[var(--bb-amber)]">
            {report.findings.map((finding, idx) => {
              const itemKey = `${turnId}-finding-${idx}`
              return (
                <li
                  key={itemKey}
                  className="text-[12.5px] leading-relaxed text-ink-secondary pl-0.5"
                >
                  <span>{finding.statement}</span>
                  {(finding.chunk_ids ?? []).map((chunkId) => {
                    const cit = citationById.get(chunkId)
                    const ticker = cit?.ticker ?? finding.ticker
                    const fiscalYear = cit?.fiscal_year ?? finding.fiscal_year
                    const sectionLabel =
                      cit?.section_title ??
                      (cit?.section_id === 'item_1a'
                        ? 'Item 1A. Risk Factors'
                        : 'Item 7. MD&A')
                    const token: ParsedInlineCitationToken = {
                      rawMatch: chunkId,
                      ticker,
                      fiscalYear,
                      sectionLabel,
                      chunkId,
                      url: cit?.sec_filing_url ?? '',
                    }
                    const isActive = isChunkActive(chunkId)
                    return (
                      <span key={`${itemKey}-cit-${chunkId}`}>
                        {' '}
                        <button
                          id={`structured-cit-${turnId}-${idx}-${chunkId}`}
                          type="button"
                          onClick={() => onSelectCitation(token)}
                          className={`citation-chip ${isActive ? 'citation-chip-active' : ''}`}
                          title={`Inspect ${ticker} FY${fiscalYear} · ${sectionLabel} in Source Viewer`}
                        >
                          <FileText
                            className="w-2.5 h-2.5 shrink-0"
                            aria-hidden="true"
                          />
                          <span>
                            {ticker} FY{String(fiscalYear).slice(-2)} ·{' '}
                            {shortCitationTopic(sectionLabel)}
                          </span>
                        </button>
                      </span>
                    )
                  })}
                  <EvidenceQuoteToggle
                    itemKey={itemKey}
                    quote={finding.evidence_quote}
                  />
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </div>
  )
}
