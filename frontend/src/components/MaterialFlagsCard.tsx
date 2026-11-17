import type { GroundedCitation, MaterialVarianceFlag } from '../types/edgar'
import type { ParsedInlineCitationToken } from '../utils/formatters'
import { citationToken } from '../utils/citationTokens'

interface MaterialFlagsCardProps {
  turnId: string
  flags: MaterialVarianceFlag[]
  totalFlags?: number
  citations: GroundedCitation[]
  onSelectCitation: (token: ParsedInlineCitationToken) => void
}

const RULE_LABELS: Record<MaterialVarianceFlag['rule'], string> = {
  yoy_change: 'YoY CHANGE',
  sign_reversal: 'SIGN FLIP',
  margin_shift: 'MARGIN SHIFT',
  period_change: 'PERIOD CHANGE',
  sector_divergence: 'VS SECTOR',
}

const METRIC_LABELS: Record<MaterialVarianceFlag['metric'], string> = {
  revenue: 'Revenue',
  operating_income: 'Operating Income',
  net_income: 'Net Income',
  research_and_development: 'R&D',
  operating_margin: 'Operating Margin',
  net_margin: 'Net Margin',
  rd_intensity: 'R&D Intensity',
}

function formatFlagValue(flag: MaterialVarianceFlag): string {
  const sign = flag.value > 0 ? '+' : ''
  switch (flag.unit) {
    case 'pct':
      return `${sign}${flag.value.toFixed(2)}%`
    case 'bps':
      return `${sign}${flag.value.toLocaleString('en-US', { maximumFractionDigits: 1 })} bps`
    case 'pp':
      return `${sign}${flag.value.toFixed(2)} pp`
    default:
      return 'Sign flip'
  }
}

/** Deterministic anomaly flags (material variances) with links to their MD&A explanation. */
export function MaterialFlagsCard({
  turnId,
  flags,
  totalFlags,
  citations,
  onSelectCitation,
}: MaterialFlagsCardProps) {
  if (!flags || flags.length === 0) return null
  const byId = new Map(citations.map((c) => [c.chunk_id, c]))

  return (
    <section
      aria-label="Material Variance Flags"
      className="mb-3 border border-hairline surface-panel overflow-hidden"
    >
      <div className="px-3 py-1.5 surface-header border-b border-hairline flex items-center justify-between gap-2 tabular-nums-mono">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-bb-amber">
          Material Variance Flags
        </span>
        <span className="text-[10px] text-ink-muted uppercase">
          {flags.length}
          {totalFlags && totalFlags > flags.length ? ` OF ${totalFlags}` : ''} · RULE-BASED
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-[11px] tabular-nums-mono">
          <thead>
            <tr className="surface-subtle border-b border-hairline text-[10px] text-ink-muted uppercase">
              <th className="py-1.5 px-2.5 font-medium">Sev</th>
              <th className="py-1.5 px-2.5 font-medium">Equity / FY</th>
              <th className="py-1.5 px-2.5 font-medium">Metric</th>
              <th className="py-1.5 px-2.5 font-medium">Rule</th>
              <th className="py-1.5 px-2.5 font-medium text-right">Change</th>
              <th className="py-1.5 px-2.5 font-medium text-right">Source</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-hairline)]">
            {flags.map((flag) => {
              const expCit = flag.explanation
                ? byId.get(flag.explanation.chunk_id)
                : undefined
              return (
                <tr
                  key={flag.flag_id}
                  id={`flag-${turnId}-${flag.flag_id}`}
                  className="hover:bg-[var(--bg-subtle)] transition-colors"
                  title={flag.description}
                >
                  <td className="py-1.5 px-2.5">
                    <span
                      className={`px-1 text-[9.5px] font-bold uppercase ${
                        flag.severity === 'high' ? 'badge-negative' : 'badge-amber'
                      }`}
                    >
                      {flag.severity === 'high' ? 'HIGH' : 'MED'}
                    </span>
                  </td>
                  <td className="py-1.5 px-2.5 whitespace-nowrap">
                    <span className="font-bold text-bb-yellow">{flag.ticker}</span>{' '}
                    <span className="text-ink-muted text-[10px]">
                      {flag.prior_fiscal_year ? `FY${String(flag.prior_fiscal_year).slice(-2)}→` : ''}
                      FY{String(flag.fiscal_year).slice(-2)}
                    </span>
                  </td>
                  <td className="py-1.5 px-2.5 text-ink-primary">
                    {METRIC_LABELS[flag.metric]}
                  </td>
                  <td className="py-1.5 px-2.5 text-ink-secondary text-[10px]">
                    {RULE_LABELS[flag.rule]}
                  </td>
                  <td
                    className={`py-1.5 px-2.5 text-right font-semibold whitespace-nowrap ${
                      flag.value > 0 ? 'text-bb-pos' : 'text-bb-neg'
                    }`}
                  >
                    {formatFlagValue(flag)}
                  </td>
                  <td className="py-1.5 px-2.5 text-right whitespace-nowrap">
                    {expCit ? (
                      <button
                        id={`btn-flag-explanation-${turnId}-${flag.flag_id}`}
                        type="button"
                        onClick={() => onSelectCitation(citationToken(expCit))}
                        className="text-bb-cyan hover:underline text-[10px] font-semibold uppercase cursor-pointer"
                        title={flag.explanation?.excerpt}
                      >
                        [MD&amp;A WHY]
                      </button>
                    ) : flag.source.sec_filing_url ? (
                      <a
                        href={flag.source.sec_filing_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-ink-muted hover:text-ink-primary text-[10px]"
                        title={`${flag.source.accession_number}${
                          flag.source.xbrl_concept ? ` · ${flag.source.xbrl_concept}` : ''
                        }`}
                      >
                        [10-K]
                      </a>
                    ) : (
                      <span className="text-ink-muted">—</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
