import { ArrowUpRight } from 'lucide-react'
import type { SearchScopeFilters } from '../types/edgar'

export interface StarterPromptPreset {
  id: string
  category: string
  title: string
  query: string
  scopePatch?: Partial<SearchScopeFilters>
}

const STARTER_PRESETS: StarterPromptPreset[] = [
  {
    id: 'peer-nvda-aapl',
    category: 'Multi-Company Peer Variance',
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
    category: 'Longitudinal Single-Issuer',
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
    category: 'Item 1A Thematic Tracking',
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
    category: 'Cross-Issuer Health Care',
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
  onSelectPreset: (preset: StarterPromptPreset) => void
}

export function StarterPrompts({ onSelectPreset }: StarterPromptsProps) {
  return (
    <section
      id="welcome-research-desk"
      aria-label="Institutional SEC Filing Research Desk"
      className="max-w-3xl mx-auto py-8 sm:py-12 px-2 animate-turn-enter"
    >
      <div className="mb-7">
        <p className="text-xs font-semibold uppercase tracking-wider text-ink-muted tabular-nums-mono mb-2">
          SEC Form 10-K Hybrid Retrieval &amp; Deterministic Variance Engine
        </p>
        <h2 className="text-xl sm:text-2xl font-semibold tracking-tight text-ink-primary leading-snug">
          Query audited financials and grounded 10-K disclosures across fiscal periods.
        </h2>
        <p className="text-sm text-ink-secondary mt-2 leading-relaxed max-w-2xl">
          Arithmetic variances (Revenue, Operating Income, Net Income, CAGR, and margin basis
          points) are computed deterministically before narrative synthesis. Every qualitative
          claim links directly to its source Item 1A or Item 7 filing passage.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {STARTER_PRESETS.map((preset) => (
          <button
            key={preset.id}
            id={`starter-prompt-${preset.id}`}
            type="button"
            onClick={() => onSelectPreset(preset)}
            className="group text-left p-4 rounded-lg surface-panel border border-hairline hover:border-strong cursor-pointer transition-all flex flex-col justify-between gap-3"
          >
            <div>
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <span className="text-[11px] font-medium text-ink-muted tabular-nums-mono uppercase tracking-wider">
                  {preset.category}
                </span>
                <ArrowUpRight
                  className="w-3.5 h-3.5 text-ink-muted group-hover:text-ink-primary transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5 shrink-0"
                  aria-hidden="true"
                />
              </div>
              <div className="text-sm font-semibold text-ink-primary leading-snug mb-1.5">
                {preset.title}
              </div>
              <p className="text-xs text-ink-secondary leading-relaxed line-clamp-2">
                &ldquo;{preset.query}&rdquo;
              </p>
            </div>
          </button>
        ))}
      </div>
    </section>
  )
}
