import type { CompanySummary, SearchScopeFilters } from '../types/edgar'
import { formatUsdCompact } from '../utils/formatters'

export interface StarterPromptPreset {
  id: string
  mnemonic: string
  category: string
  title: string
  query: string
  scopePatch?: Partial<SearchScopeFilters>
}

const STARTER_PRESETS: StarterPromptPreset[] = [
  {
    id: 'peer-nvda-aapl',
    mnemonic: 'COMP',
    category: 'PEER VARIANCE',
    title: 'NVIDIA vs. Apple: FY2024–FY2026 Revenue, Margin & AI Drivers',
    query:
      "Compare NVIDIA and Apple's revenue, operating margin, and AI infrastructure & R&D commentary from 2024 to 2026.",
    scopePatch: {
      selectedTickers: ['NVDA', 'AAPL'],
      gicsSector: 'ALL',
      sectionId: 'ALL',
      startYear: 2024,
      endYear: 2026,
    },
  },
  {
    id: 'longitudinal-msft',
    mnemonic: 'FA',
    category: 'SINGLE ISSUER',
    title: 'Microsoft FY2023–FY2025 Cloud Scale & Operating Leverage',
    query:
      "Analyze Microsoft's period-over-period revenue, operating income, and Azure AI CapEx drivers from 2023 to 2025.",
    scopePatch: {
      selectedTickers: ['MSFT'],
      gicsSector: 'ALL',
      sectionId: 'ALL',
      startYear: 2023,
      endYear: 2025,
    },
  },
  {
    id: 'thematic-export-risk',
    mnemonic: 'RISK',
    category: 'ITEM 1A TRACKER',
    title: 'Export Controls, China Licensing & AI Supply Chain Risk Evolution',
    query:
      'How has the wording on export controls, China trade restrictions, and AI supply chain risks evolved across Information Technology Item 1A filings from 2024 to 2026?',
    scopePatch: {
      selectedTickers: [],
      gicsSector: 'Information Technology',
      sectionId: 'item_1a',
      startYear: 2024,
      endYear: 2026,
    },
  },
  {
    id: 'peer-healthcare-lly-unh',
    mnemonic: 'PEER',
    category: 'HEALTH CARE',
    title: 'Eli Lilly vs. UnitedHealth: Margin Trajectory & Regulatory Disclosures',
    query:
      'Compare Eli Lilly and UnitedHealth financial variances, operating margins, and Item 1A regulatory risk factors from 2023 to 2025.',
    scopePatch: {
      selectedTickers: ['LLY', 'UNH'],
      gicsSector: 'Health Care',
      sectionId: 'ALL',
      startYear: 2023,
      endYear: 2025,
    },
  },
]

interface StarterPromptsProps {
  companies: CompanySummary[]
  onSelectPreset: (preset: StarterPromptPreset) => void
  onSelectIssuerFiling?: (ticker: string, fiscalYear: number) => void
}

export function StarterPrompts({
  companies,
  onSelectPreset,
  onSelectIssuerFiling,
}: StarterPromptsProps) {
  return (
    <section
      id="welcome-research-desk"
      aria-label="Institutional SEC Filing Research Desk"
      className="py-4 space-y-5"
    >
      {/* Terminal Function Header */}
      <div className="border border-hairline surface-panel">
        <div className="px-3 py-1.5 surface-header border-b border-hairline flex items-center justify-between gap-2 tabular-nums-mono">
          <span className="text-[11px] font-bold text-bb-amber uppercase tracking-wider">
            EDGR &lt;GO&gt; // XBRL VARIANCE &amp; 10-K HYBRID RETRIEVAL
          </span>
          <span className="text-[10px] text-ink-muted uppercase hidden sm:inline">
            ITEM 1A + ITEM 7 GROUNDED
          </span>
        </div>
        <div className="px-3 py-2.5 flex flex-col gap-1">
          <h2 className="text-sm font-semibold text-ink-primary">
            Query audited financials and grounded 10-K disclosures across fiscal periods.
          </h2>
          <p className="text-xs text-ink-secondary">
            Revenue, Operating Income, Net Income, CAGR, and margin basis points are
            calculated deterministically from XBRL data before narrative synthesis.
          </p>
        </div>
      </div>

      {/* Preset Analytical Screens Directory */}
      <div className="border border-hairline surface-panel">
        <div className="px-3 py-1.5 surface-header border-b border-hairline flex items-center justify-between text-[11px] tabular-nums-mono uppercase">
          <span className="font-semibold text-ink-primary">
            QUICK ANALYTICAL FUNCTIONS
          </span>
          <span className="text-ink-muted">SELECT TO EXECUTE</span>
        </div>

        <div className="divide-y divide-[var(--border-hairline)]">
          {STARTER_PRESETS.map((preset, index) => (
            <button
              key={preset.id}
              id={`starter-prompt-${preset.id}`}
              type="button"
              onClick={() => onSelectPreset(preset)}
              className="w-full text-left px-3 py-2.5 hover:bg-[var(--bg-subtle)] cursor-pointer transition-colors flex items-start justify-between gap-3 group"
            >
              <div className="flex items-start gap-2.5 min-w-0">
                <span className="tabular-nums-mono text-[11px] font-bold text-bb-amber shrink-0 pt-0.5">
                  {index + 1}&gt;
                </span>
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="tabular-nums-mono text-[10px] font-bold px-1.5 py-0.2 badge-amber">
                      {preset.mnemonic}
                    </span>
                    <span className="tabular-nums-mono text-[10px] text-ink-muted uppercase">
                      {preset.category}
                    </span>
                    <span className="text-xs font-semibold text-ink-primary group-hover:text-[var(--bb-yellow)] transition-colors">
                      {preset.title}
                    </span>
                  </div>
                  <p className="text-[11px] text-ink-secondary mt-0.5 truncate">
                    {preset.query}
                  </p>
                </div>
              </div>
              <span className="tabular-nums-mono text-[10px] text-bb-amber opacity-75 group-hover:opacity-100 shrink-0 pt-0.5">
                &lt;GO&gt;
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Covered S&P 500 Financial Universe Table */}
      {companies.length > 0 && (
        <div className="border border-hairline surface-panel overflow-hidden">
          <div className="px-3 py-1.5 surface-header border-b border-hairline flex items-center justify-between text-[11px] tabular-nums-mono uppercase">
            <span className="font-semibold text-ink-primary">
              S&amp;P 500 COVERAGE MONITOR ({companies.length} EQUITIES)
            </span>
            <span className="text-ink-muted">LATEST AUDITED 10-K · USD</span>
          </div>

          <div className="overflow-x-auto max-h-72 overflow-y-auto">
            <table className="w-full text-left border-collapse text-[11px] tabular-nums-mono">
              <thead className="sticky top-0 surface-subtle border-b border-hairline text-[10px] text-ink-muted uppercase">
                <tr>
                  <th className="py-1.5 px-2.5 font-medium">Ticker</th>
                  <th className="py-1.5 px-2.5 font-medium">Issuer</th>
                  <th className="py-1.5 px-2.5 font-medium hidden md:table-cell">
                    Sector
                  </th>
                  <th className="py-1.5 px-2.5 font-medium text-right">FY</th>
                  <th className="py-1.5 px-2.5 font-medium text-right">
                    Revenue
                  </th>
                  <th className="py-1.5 px-2.5 font-medium text-right">
                    Op Mgn
                  </th>
                  <th className="py-1.5 px-2.5 font-medium text-right">
                    Filing
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-hairline)]">
                {companies.map((c) => (
                  <tr
                    key={c.ticker}
                    className="hover:bg-[var(--bg-subtle)] even:bg-white/[0.012] transition-colors"
                  >
                    <td className="py-1.5 px-2.5 font-bold text-bb-yellow">
                      {c.ticker}
                    </td>
                    <td className="py-1.5 px-2.5 text-ink-primary truncate max-w-[160px] font-sans">
                      {c.company_name}
                    </td>
                    <td className="py-1.5 px-2.5 text-ink-muted truncate max-w-[150px] hidden md:table-cell font-sans">
                      {c.gics_sector}
                    </td>
                    <td className="py-1.5 px-2.5 text-right text-ink-secondary">
                      FY{String(c.latest_fiscal_year).slice(-2)}
                    </td>
                    <td className="py-1.5 px-2.5 text-right text-ink-primary font-medium">
                      {formatUsdCompact(c.latest_revenue)}
                    </td>
                    <td className="py-1.5 px-2.5 text-right text-bb-pos">
                      {c.latest_operating_margin_pct !== null
                        ? `${c.latest_operating_margin_pct.toFixed(1)}%`
                        : '—'}
                    </td>
                    <td className="py-1.5 px-2.5 text-right">
                      {onSelectIssuerFiling && (
                        <button
                          id={`btn-monitor-10k-${c.ticker}`}
                          type="button"
                          onClick={() =>
                            onSelectIssuerFiling(c.ticker, c.latest_fiscal_year)
                          }
                          className="text-bb-cyan hover:underline cursor-pointer uppercase text-[10px] font-semibold"
                        >
                          [10-K]
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  )
}
