import type {
  ChatTurnResponse,
  CompanyVarianceReport,
  DisclosureDeltaResult,
  MaterialVarianceFlag,
} from '../types/edgar'

/**
 * Render predicates for the deterministic evidence cards.
 *
 * The FinancialAnalystAgent invokes deterministic engines as tools, so each
 * payload is only populated when the agent actually called the matching tool:
 * - variance_reports         ← calculate_period_variances
 * - material_variance_flags  ← flag_material_variances
 * - disclosure_deltas        ← compare_disclosures_across_years / sector map-reduce
 * Empty arrays / null payloads mean "tool not called" and must render nothing.
 */

export function getRenderableVarianceReports(
  resp: Pick<ChatTurnResponse, 'variance_reports'> | null | undefined
): CompanyVarianceReport[] {
  const reports = resp?.variance_reports
  return Array.isArray(reports) ? reports : []
}

export function getRenderableMaterialFlags(
  resp: Pick<ChatTurnResponse, 'material_variance_flags'> | null | undefined
): MaterialVarianceFlag[] {
  const flags = resp?.material_variance_flags
  return Array.isArray(flags) ? flags : []
}

/** True only for a disclosure-delta payload the engine actually produced. */
export function isRenderableDisclosureDeltas(
  result: DisclosureDeltaResult | null | undefined
): result is DisclosureDeltaResult & {
  companies: NonNullable<DisclosureDeltaResult['companies']>
} {
  return (
    result != null &&
    typeof result === 'object' &&
    result.status === 'success' &&
    Array.isArray(result.companies) &&
    result.companies.length > 0
  )
}
