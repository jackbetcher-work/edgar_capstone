/**
 * Human-readable labels for XBRL concepts and 10-K section breadcrumbs.
 *
 * Inline citations carry raw provenance strings such as
 * `us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest`
 * or `edgartools:Financials.get_operating_income (us-gaap:OperatingIncomeLoss)`.
 * Analysts need the financial meaning ("Pre-tax Income"); the raw concept stays
 * available in tooltips and the XBRL Fact inspector for verification.
 */

const EXACT_CONCEPT_LABELS: Record<string, string> = {
  Revenues: 'Revenue',
  RevenueFromContractWithCustomerExcludingAssessedTax: 'Revenue',
  RevenueFromContractWithCustomerIncludingAssessedTax: 'Revenue',
  RegulatedAndUnregulatedOperatingRevenue: 'Revenue',
  RegulatedOperatingRevenue: 'Regulated Revenue',
  RevenuesNetOfInterestExpense: 'Net Revenue',
  RevenuesAndOther: 'Revenue',
  InterestAndDividendIncomeOperating: 'Interest & Dividend Income',
  OperatingLeaseLeaseIncome: 'Lease Income',
  OperatingIncomeLoss: 'Operating Income',
  IncomeBeforeInterestAndIncomeTaxes: 'EBIT',
  NetIncomeLoss: 'Net Income',
  ProfitLoss: 'Net Income (incl. NCI)',
  NetIncomeLossAvailableToCommonStockholdersBasic: 'Net Income to Common',
  IncomeLossAttributableToParent: 'Net Income to Parent',
  IncomeLossIncludingPortionAttributableToNoncontrollingInterest:
    'Net Income (incl. NCI)',
  IncomeLossFromContinuingOperationsBeforeNoncontrollingInterest:
    'Income from Continuing Ops',
  ResearchAndDevelopmentExpense: 'R&D Expense',
  ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost: 'R&D Expense',
  ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost:
    'R&D Expense (Software)',
  GrossProfit: 'Gross Profit',
  CostOfRevenue: 'Cost of Revenue',
  EarningsPerShareBasic: 'EPS (Basic)',
  EarningsPerShareDiluted: 'EPS (Diluted)',
  Assets: 'Total Assets',
  Liabilities: 'Total Liabilities',
  StockholdersEquity: "Shareholders' Equity",
  NetCashProvidedByUsedInOperatingActivities: 'Operating Cash Flow',
}

/** Pattern fallbacks for issuer-extension and long-tail concepts. */
const PATTERN_CONCEPT_LABELS: Array<[RegExp, string]> = [
  [/BeforeIncomeTaxes|BeforeEarningsFromAffiliatesIncomeTaxes/i, 'Pre-tax Income'],
  [/ResearchAndDevelopment/i, 'R&D Expense'],
  [/^NetIncome/i, 'Net Income'],
  [/OperatingIncome/i, 'Operating Income'],
  [/Revenue/i, 'Revenue'],
]

/** Labels for `edgartools:Financials.get_<metric>` accessor provenance strings. */
const ACCESSOR_LABELS: Record<string, string> = {
  revenue: 'Revenue',
  operating_income: 'Operating Income',
  net_income: 'Net Income',
  research_and_development: 'R&D Expense',
  gross_profit: 'Gross Profit',
}

const ACRONYMS = new Set(['eps', 'ebit', 'ebitda', 'nci', 'r&d'])

function titleCaseWords(words: string[]): string {
  return words
    .map((w) =>
      ACRONYMS.has(w.toLowerCase())
        ? w.toUpperCase()
        : w.charAt(0).toUpperCase() + w.slice(1)
    )
    .join(' ')
}

/**
 * Reduce a provenance string to its most specific bare concept name:
 * `edgartools:Financials.get_x (us-gaap:Foo)` → `Foo`,
 * `sec-companyfacts:us-gaap:Foo` → `Foo`, `us-gaap:Foo` → `Foo`.
 */
export function bareXbrlConceptName(raw: string): string {
  const trimmed = (raw ?? '').trim()
  const parenthetical = /\(([^)]+)\)\s*$/.exec(trimmed)
  const core = parenthetical ? parenthetical[1].trim() : trimmed
  const afterNamespace = core.includes(':')
    ? core.slice(core.lastIndexOf(':') + 1)
    : core
  return afterNamespace.trim()
}

/** Map a raw XBRL concept / provenance string to a short analyst-facing label. */
export function humanizeXbrlConcept(raw: string | null | undefined): string {
  if (!raw || !raw.trim()) return 'Reported Figure'
  const bare = bareXbrlConceptName(raw)

  const accessor = /^Financials\.get_([a-z_]+)$/i.exec(bare)
  if (accessor) {
    const key = accessor[1].toLowerCase()
    return ACCESSOR_LABELS[key] ?? titleCaseWords(key.split('_'))
  }

  if (EXACT_CONCEPT_LABELS[bare]) return EXACT_CONCEPT_LABELS[bare]

  for (const [pattern, label] of PATTERN_CONCEPT_LABELS) {
    if (pattern.test(bare)) return label
  }

  // Generic CamelCase → words, capped so chips stay compact.
  const words = bare
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .split(/[\s_.]+/)
    .filter(Boolean)
  if (words.length === 0) return 'Reported Figure'
  const label = titleCaseWords(words.slice(0, 4))
  return words.length > 4 ? `${label}…` : label
}

export interface SectionLabelParts {
  /** Short section name, e.g. "MD&A" or "Risk Factors". */
  section: string
  /** Subsection / topic breadcrumb after the em dash, when present. */
  topic: string | null
}

/**
 * Split a section breadcrumb such as `Item 7. MD&A — Gross Margin` or
 * `Item 1A Risk Factors` into a short section name and optional topic.
 */
export function describeSectionLabel(label: string | null | undefined): SectionLabelParts {
  const text = (label ?? '').trim()
  const [head, ...rest] = text.split(/\s+[—–]\s+/)
  const topicRaw = rest.join(' — ').trim()
  const lowerHead = (head ?? '').toLowerCase()
  const section =
    lowerHead.includes('1a') || lowerHead.includes('risk')
      ? 'Risk Factors'
      : lowerHead.includes('7') || lowerHead.includes('md&a') || lowerHead.includes('management')
        ? 'MD&A'
        : head || '10-K'
  return { section, topic: topicRaw || null }
}

/** Max topic length that still reads well inside an inline citation chip. */
const MAX_CHIP_TOPIC_CHARS = 30

/** Compact chip text for a 10-K passage citation: topic when short, else section name. */
export function shortCitationTopic(sectionLabel: string): string {
  const { section, topic } = describeSectionLabel(sectionLabel)
  if (topic && topic.length <= MAX_CHIP_TOPIC_CHARS) {
    return topic
  }
  return section
}
