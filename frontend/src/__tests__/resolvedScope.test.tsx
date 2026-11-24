import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { ChatTurnItem } from '../components/ChatTurnItem'
import type { ChatTurnResponse, ConversationTurn, ResolvedQueryScope } from '../types/edgar'
import { summarizeResolvedScope } from '../utils/resolvedScope'

describe('summarizeResolvedScope', () => {
  it('describes tickers, fiscal window and sections', () => {
    expect(
      summarizeResolvedScope({
        tickers: ['AAPL', 'MSFT'],
        gics_sector: 'ALL',
        section_id: 'item_7',
        start_year: 2022,
        end_year: 2024,
      })
    ).toEqual({
      issuers: 'AAPL, MSFT',
      issuersFull: 'AAPL, MSFT',
      years: 'FY2022–FY2024',
      sections: 'MD&A',
      carriedOver: false,
    })
  })

  it('falls back to sector or all companies, abbreviates long ticker lists, and flags memory', () => {
    expect(
      summarizeResolvedScope({
        tickers: [],
        gics_sector: 'Information Technology',
        section_id: 'item_1a',
        start_year: 2025,
        end_year: 2025,
      })
    ).toMatchObject({ issuers: 'Information Technology', years: 'FY2025', sections: 'Risk Factors' })

    expect(
      summarizeResolvedScope({
        tickers: [],
        gics_sector: 'ALL',
        section_id: 'ALL',
        start_year: 2024,
        end_year: 2026,
      })
    ).toMatchObject({ issuers: 'All companies', sections: 'MD&A + Risk Factors' })

    expect(
      summarizeResolvedScope({
        tickers: ['A', 'B', 'C', 'D', 'E', 'F'],
        gics_sector: 'ALL',
        section_id: 'ALL',
        start_year: 2023,
        end_year: 2024,
        memory_inherited: true,
      })
    ).toMatchObject({ issuers: 'A, B, C, D +2', issuersFull: 'A, B, C, D, E, F', carriedOver: true })

    expect(
      summarizeResolvedScope({
        tickers: ['GD', 'LMT', 'NOC', 'RTX'],
        gics_sector: 'ALL',
        gics_sub_industry: 'Aerospace & Defense',
        section_id: 'ALL',
        start_year: 2024,
        end_year: 2026,
      })
    ).toMatchObject({
      issuers: 'Aerospace & Defense (4)',
      issuersFull: 'Aerospace & Defense: GD, LMT, NOC, RTX',
      carriedOver: false,
    })

    expect(summarizeResolvedScope(null)).toBeNull()
  })
})

describe('ChatTurnItem interpreted scope', () => {
  afterEach(() => cleanup())

  function turn(status: ChatTurnResponse['status'], scope: ResolvedQueryScope): ConversationTurn {
    return {
      id: 'turn-scope',
      createdAt: '2026-11-19T00:00:00Z',
      userQuery: 'And how did that compare with Microsoft?',
      isLoading: false,
      response: {
        status,
        model_armor_blocked: false,
        intent: status === 'clarification' ? 'clarification_needed' : 'multi_company_peer_comparison',
        orchestrator: 'RootOrchestrator',
        resolved_scope: scope,
        answer_markdown: '### Answer\n\nText.',
        citations: [],
        context_cache: null,
        fact_check: {
          passed: true,
          checked_metrics_count: 0,
          matched_citations_count: 0,
          has_required_citations: true,
          discrepancies: [],
          model_latency_ms: 1,
        },
        agent_trace: [],
        total_latency_ms: 100,
      },
    }
  }

  function renderTurn(t: ConversationTurn) {
    render(
      <ChatTurnItem
        turn={t}
        activeChunkId={null}
        onSelectInlineCitation={vi.fn()}
        onFocusTurnCitations={vi.fn()}
        onInspectFilingSection={vi.fn()}
        onSubmitFeedback={vi.fn()}
      />
    )
  }

  const SCOPE: ResolvedQueryScope = {
    tickers: ['AAPL', 'MSFT'],
    gics_sector: 'ALL',
    section_id: 'ALL',
    start_year: 2023,
    end_year: 2024,
    memory_inherited: true,
  }

  it('shows the resolved scope and marks scope carried over from earlier questions', () => {
    renderTurn(turn('success', SCOPE))
    const scope = screen.getByLabelText('Interpreted scope')
    expect(scope.textContent).toContain('AAPL, MSFT')
    expect(scope.textContent).toContain('FY2023–FY2024')
    expect(scope.textContent).toContain('carried over')
    expect(scope.getAttribute('title')).toMatch(/Rephrase the question to change it/)
  })

  it('omits the scope line for clarification turns', () => {
    renderTurn(turn('clarification', SCOPE))
    expect(screen.queryByLabelText('Interpreted scope')).toBeNull()
  })
})
