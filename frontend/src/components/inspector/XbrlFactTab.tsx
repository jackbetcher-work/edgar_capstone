import type { ReactNode } from 'react'
import { ExternalLink } from 'lucide-react'
import type { XbrlFactSelection } from '../../types/edgar'
import { formatUsdCompact } from '../../utils/formatters'

interface XbrlFactTabProps {
  selectedXbrlFact: XbrlFactSelection | null
  onOpenFilingSection: (
    ticker: string,
    fiscalYear: number,
    sectionId: 'item_7' | 'item_1a'
  ) => void
}

function ProvenanceRow({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <div className="px-2.5 py-1.5 flex items-center justify-between gap-2">
      <span className="text-ink-muted uppercase text-[10px]">{label}</span>
      {children}
    </div>
  )
}

export function XbrlFactTab({
  selectedXbrlFact,
  onOpenFilingSection,
}: XbrlFactTabProps) {
  return (
    <div className="flex-1 overflow-y-auto p-3.5 surface-filing-sheet tabular-nums-mono">
      {selectedXbrlFact ? (
        <div id="xbrl-fact-inspector" className="space-y-3">
          <div className="pb-2.5 border-b border-hairline">
            <div className="flex items-center justify-between gap-2 mb-1">
              <span className="text-[10px] uppercase tracking-wider font-bold text-bb-amber">
                SEC XBRL NUMERIC PROVENANCE
              </span>
              <span className="px-1.5 py-0.2 text-[9.5px] uppercase font-semibold badge-accent">
                AUDITED 10-K FACT
              </span>
            </div>
            <h3 className="text-sm font-semibold text-ink-primary">
              {selectedXbrlFact.title}
            </h3>
            {selectedXbrlFact.companyName && (
              <p className="text-[11px] text-ink-secondary font-sans mt-0.5">
                {selectedXbrlFact.companyName} ({selectedXbrlFact.ticker})
              </p>
            )}
          </div>

          {(selectedXbrlFact.valueFormatted ||
            selectedXbrlFact.deltaFormatted) && (
            <div className="grid grid-cols-2 gap-2">
              {selectedXbrlFact.valueFormatted && (
                <div className="p-2 border border-hairline surface-subtle">
                  <div className="text-[9.5px] uppercase text-ink-muted">
                    REPORTED VALUE
                  </div>
                  <div className="text-xs font-semibold text-ink-primary mt-0.5">
                    {selectedXbrlFact.valueFormatted}
                  </div>
                </div>
              )}
              {selectedXbrlFact.deltaFormatted && (
                <div className="p-2 border border-hairline surface-subtle">
                  <div className="text-[9.5px] uppercase text-ink-muted">
                    COMPUTED VARIANCE
                  </div>
                  <div className="text-xs font-semibold text-bb-amber mt-0.5">
                    {selectedXbrlFact.deltaFormatted}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="border border-hairline surface-panel divide-y divide-[var(--border-hairline)] text-[11px]">
            <ProvenanceRow label="Metric">
              <span className="text-ink-primary font-medium">
                {selectedXbrlFact.metricLabel}
              </span>
            </ProvenanceRow>
            <ProvenanceRow label="Fiscal Year / Period End">
              <span className="text-ink-primary">
                FY{selectedXbrlFact.fiscalYear}
                {selectedXbrlFact.periodOfReport
                  ? ` (${selectedXbrlFact.periodOfReport})`
                  : ''}
              </span>
            </ProvenanceRow>
            {selectedXbrlFact.filingDate && (
              <ProvenanceRow label="SEC Filing Date">
                <span className="text-ink-primary">
                  {selectedXbrlFact.filingDate}
                </span>
              </ProvenanceRow>
            )}
            {selectedXbrlFact.xbrlConcept && (
              <ProvenanceRow label="XBRL Concept">
                <code className="text-bb-cyan text-[10.5px] break-all">
                  {selectedXbrlFact.xbrlConcept}
                </code>
              </ProvenanceRow>
            )}
            {selectedXbrlFact.accessionNumber && (
              <ProvenanceRow label="Accession Number">
                <code className="text-ink-primary text-[10.5px]">
                  {selectedXbrlFact.accessionNumber}
                </code>
              </ProvenanceRow>
            )}
            {selectedXbrlFact.denominatorXbrlConcept && (
              <ProvenanceRow label="Denominator Concept">
                <code className="text-bb-cyan text-[10.5px] break-all">
                  {selectedXbrlFact.denominatorXbrlConcept}
                </code>
              </ProvenanceRow>
            )}
            {selectedXbrlFact.priorFiscalYear && (
              <ProvenanceRow label="Prior Period">
                <span className="text-ink-secondary">
                  FY{selectedXbrlFact.priorFiscalYear}
                  {selectedXbrlFact.priorPeriodOfReport
                    ? ` (${selectedXbrlFact.priorPeriodOfReport})`
                    : ''}
                </span>
              </ProvenanceRow>
            )}
            {selectedXbrlFact.priorXbrlConcept && (
              <ProvenanceRow label="Prior XBRL Concept">
                <code className="text-ink-secondary text-[10.5px] break-all">
                  {selectedXbrlFact.priorXbrlConcept}
                </code>
              </ProvenanceRow>
            )}
            {selectedXbrlFact.priorAccessionNumber && (
              <ProvenanceRow label="Prior Accession Number">
                <code className="text-ink-secondary text-[10.5px]">
                  {selectedXbrlFact.priorAccessionNumber}
                </code>
              </ProvenanceRow>
            )}
          </div>

          {selectedXbrlFact.derivedFrom &&
            selectedXbrlFact.derivedFrom.length > 0 && (
              <div className="border border-hairline surface-panel overflow-hidden">
                <div className="px-2.5 py-1 surface-subtle border-b border-hairline text-[10px] uppercase text-ink-muted font-semibold">
                  DERIVED CALCULATION INPUTS
                </div>
                <div className="divide-y divide-[var(--border-hairline)] text-[10.5px]">
                  {selectedXbrlFact.derivedFrom.map((inp, idx) => (
                    <div
                      key={`${inp.ticker}-${inp.fiscal_year}-${inp.metric}-${idx}`}
                      className="px-2.5 py-1.5 space-y-0.5"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-bb-yellow font-semibold">
                          {inp.ticker} FY{inp.fiscal_year} · {inp.metric}
                        </span>
                        <span className="text-ink-primary font-semibold">
                          {inp.value !== undefined && inp.value !== null
                            ? formatUsdCompact(inp.value)
                            : '—'}
                        </span>
                      </div>
                      <div className="text-[9.5px] text-ink-muted flex flex-wrap gap-x-2">
                        {inp.xbrl_concept && <span>{inp.xbrl_concept}</span>}
                        {inp.accession_number && (
                          <span>acc:{inp.accession_number}</span>
                        )}
                        {inp.period_of_report && (
                          <span>ended {inp.period_of_report}</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

          <div className="flex flex-wrap items-center gap-2 pt-1">
            <button
              id="btn-xbrl-open-mdna"
              type="button"
              onClick={() =>
                onOpenFilingSection(
                  selectedXbrlFact.ticker,
                  selectedXbrlFact.fiscalYear,
                  'item_7'
                )
              }
              className="px-2.5 py-1 text-[10.5px] uppercase btn-terminal-key cursor-pointer"
            >
              Open FY{selectedXbrlFact.fiscalYear} Item 7 MD&amp;A
            </button>
            {selectedXbrlFact.secFilingUrl && (
              <a
                href={selectedXbrlFact.secFilingUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 px-2.5 py-1 text-[10.5px] uppercase text-bb-cyan border border-hairline hover:underline"
              >
                <span>SEC EDGAR FY{selectedXbrlFact.fiscalYear}</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            )}
            {selectedXbrlFact.priorSecFilingUrl &&
              selectedXbrlFact.priorFiscalYear && (
                <a
                  href={selectedXbrlFact.priorSecFilingUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 px-2.5 py-1 text-[10.5px] uppercase text-ink-secondary border border-hairline hover:text-ink-primary"
                >
                  <span>SEC EDGAR FY{selectedXbrlFact.priorFiscalYear}</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              )}
          </div>
        </div>
      ) : (
        <p className="text-xs text-ink-muted text-center py-12 uppercase">
          CLICK ANY [XBRL] TAG OR NUMERIC CITATION TO INSPECT PROVENANCE.
        </p>
      )}
    </div>
  )
}
