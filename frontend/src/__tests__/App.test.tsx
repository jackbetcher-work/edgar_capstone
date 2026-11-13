import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from '../App'

describe('EDGAR <GO> Financial Terminal Workspace', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('offline test mode'))
  })

  afterEach(() => {
    cleanup()
  })

  it('renders single h1 heading, terminal dark theme, scope bar, coverage monitor, and 10-K pane', async () => {
    render(<App />)

    const headings = screen.getAllByRole('heading', { level: 1 })
    expect(headings).toHaveLength(1)
    expect(headings[0].textContent).toBe('EDGAR Analyst')

    expect(document.documentElement.getAttribute('data-theme')).toBe('terminal')

    expect(
      await screen.findByText(/Query audited financials and grounded 10-K disclosures/i)
    ).toBeTruthy()

    expect(screen.getByLabelText(/GICS Sector/i)).toBeTruthy()
    expect(screen.getByLabelText(/Form 10-K Section/i)).toBeTruthy()
    expect(screen.getByText(/Grounded Chunks/i)).toBeTruthy()
    expect(screen.getByText(/S&P 500 COVERAGE MONITOR \(18 EQUITIES\)/i)).toBeTruthy()
  })

  it('executes a peer comparison RAG query, renders variance math, inline citations, and opens 10-K section', async () => {
    render(<App />)

    const starterBtn = await screen.findByText(
      /NVIDIA vs\. Apple: FY2024–FY2026 Revenue, Margin & AI Drivers/i
    )
    fireEvent.click(starterBtn)

    await waitFor(() => {
      expect(screen.getByText(/Fact-Check Verified/i)).toBeTruthy()
    })

    // Deterministic Variance Table rendered
    expect(
      screen.getByText(/Deterministic Variance Calculation Engine/i)
    ).toBeTruthy()
    expect(screen.getAllByText(/\+254\.45%/i).length).toBeGreaterThan(0)

    // Inline citation chip rendered and interactive
    const inlineChips = await screen.findAllByTitle(/Inspect NVDA FY2026/i)
    expect(inlineChips.length).toBeGreaterThan(0)
    fireEvent.click(inlineChips[0])

    // Open in Full 10-K Section reader
    const openSectionButtons = await screen.findAllByText(/Open in 10-K Section/i)
    fireEvent.click(openSectionButtons[0])

    await waitFor(() => {
      expect(screen.getByText(/UNITED STATES SEC · FORM 10-K/i)).toBeTruthy()
    })
  })

  it('focuses the EDGR> command line via / shortcut and executes a query with <GO>', async () => {
    render(<App />)
    await screen.findByText(/EDGR <GO>/i)

    fireEvent.keyDown(window, { key: '/' })
    const cmdInput = screen.getByLabelText(
      /Ask a natural language financial or SEC 10-K question/i
    )
    expect(document.activeElement).toBe(cmdInput)

    fireEvent.change(cmdInput, {
      target: { value: 'NVDA FY2024 to FY2026 revenue variance' },
    })
    fireEvent.click(screen.getByRole('button', { name: /^<GO>$/ }))

    await waitFor(() => {
      expect(screen.getByText(/CMD>/i)).toBeTruthy()
      expect(
        screen.getByText('NVDA FY2024 to FY2026 revenue variance')
      ).toBeTruthy()
    })
  })
})
