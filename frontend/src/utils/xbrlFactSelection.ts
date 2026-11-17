import type {
  AnnualStepVariance,
  CompanyVarianceReport,
  MaterialVarianceFlag,
  MetricVariance,
  XbrlFactSelection,
} from '../types/edgar'
import {
  formatSignedDeltaUsd,
  formatSignedPercent,
  formatUsdCompact,
  type ParsedInlineXbrlToken,
} from './formatters'

/**
 * Shared builders for `XbrlFactSelection` payloads passed to the XBRL Fact
 * Provenance inspector tab across `VarianceTableCard`, `MaterialFlagsCard`,
 * and inline XBRL citation chips in `App`.
 */

export function buildXbrlFactFromVariance(
  rep: CompanyVarianceReport,
  mv: MetricVariance,
  metricLabel: string
): XbrlFactSelection {
  const startYear = mv.start_year || rep.start_year
  const endYear = mv.end_year || rep.end_year
  const startKey = String(startYear)
  const endKey = String(endYear)

  return {
    title: `${rep.ticker} ${metricLabel} (FY${mv.start_year}→FY${mv.end_year})`,
    ticker: rep.ticker,
    companyName: rep.company_name,
    fiscalYear: endYear,
    priorFiscalYear: startYear,
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
    secFilingUrl: mv.end_sec_filing_url ?? rep.source_urls?.[endKey] ?? null,
    priorSecFilingUrl:
      mv.start_sec_filing_url ?? rep.source_urls?.[startKey] ?? null,
    derivedFrom: mv.derived_from,
  }
}

export function buildXbrlFactFromAnnualStep(
  rep: CompanyVarianceReport,
  step: AnnualStepVariance
): XbrlFactSelection {
  const fyKey = String(step.fiscal_year)
  const periodOfReport =
    step.period_of_report ?? rep.period_end_dates?.[fyKey] ?? null
  const accessionNumber =
    step.accession_number ?? rep.accession_numbers?.[fyKey] ?? null
  const xbrlConcept =
    step.metric_provenance?.revenue ??
    rep.metric_provenance?.[fyKey]?.revenue ??
    null

  return {
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
    periodOfReport,
    priorPeriodOfReport: step.prior_period_of_report ?? null,
    filingDate: step.filing_date ?? rep.filing_dates?.[fyKey] ?? null,
    accessionNumber,
    priorAccessionNumber: step.prior_accession_number ?? null,
    xbrlConcept,
    secFilingUrl: step.sec_filing_url,
    priorSecFilingUrl: step.prior_sec_filing_url ?? null,
  }
}

export function buildXbrlFactFromFlag(
  flag: MaterialVarianceFlag,
  metricLabel: string,
  ruleLabel: string,
  deltaFormatted: string
): XbrlFactSelection {
  return {
    title: `${flag.ticker} ${metricLabel} (${ruleLabel})`,
    ticker: flag.ticker,
    companyName: flag.company_name,
    fiscalYear: flag.fiscal_year,
    priorFiscalYear: flag.prior_fiscal_year,
    metricLabel,
    deltaFormatted,
    periodOfReport: flag.source.period_of_report,
    priorPeriodOfReport: flag.source.prior_period_of_report,
    accessionNumber: flag.source.accession_number,
    priorAccessionNumber: flag.source.prior_accession_number,
    xbrlConcept: flag.source.xbrl_concept,
    priorXbrlConcept: flag.source.prior_xbrl_concept ?? null,
    denominatorXbrlConcept: flag.source.denominator_xbrl_concept ?? null,
    priorDenominatorXbrlConcept:
      flag.source.prior_denominator_xbrl_concept ?? null,
    secFilingUrl: flag.source.sec_filing_url,
    priorSecFilingUrl: flag.source.prior_sec_filing_url,
  }
}

export function buildXbrlFactFromInlineToken(
  token: ParsedInlineXbrlToken,
  matchedRep?: CompanyVarianceReport
): XbrlFactSelection {
  const fyKey = String(token.fiscalYear)
  return {
    title: `${token.ticker} FY${token.fiscalYear} · ${token.xbrlConcept}`,
    ticker: token.ticker,
    companyName: matchedRep?.company_name,
    fiscalYear: token.fiscalYear,
    metricLabel: token.xbrlConcept,
    periodOfReport: matchedRep?.period_end_dates?.[fyKey] ?? null,
    filingDate: matchedRep?.filing_dates?.[fyKey] ?? null,
    accessionNumber: token.accessionNumber,
    xbrlConcept: token.xbrlConcept,
    secFilingUrl: token.url || matchedRep?.source_urls?.[fyKey] || null,
  }
}
