import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from '../App'

describe('EDGAR Analyst Source-Grounded Hybrid RAG Workspace', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('offline test mode'))
  })

  afterEach(() => {
    cleanup()
  })

  it('renders single h1 heading, hybrid scope filters, starter prompts, and source viewer', async () => {
    render(<App />)

    const headings = screen.getAllByRole('heading', { level: 1 })
    expect(headings).toHaveLength(1)
    expect(headings[0].textContent).toBe('EDGAR Analyst')

    expect(
      await screen.findByText(/Query audited financials and grounded 10-K disclosures/i)
    ).toBeTruthy()

    expect(screen.getByLabelText(/GICS Sector/i)).toBeTruthy()
    expect(screen.getByLabelText(/Form 10-K Section/i)).toBeTruthy()
    expect(screen.getByText(/Grounded Chunks/i)).toBeTruthy()
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

  it('toggles between light warm paper and dark desk themes', async () => {
    render(<App />)
    await screen.findByText(/Local Corpus Snapshot/i)
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')

    const toggleBtn = screen.getByRole('button', {
      name: /Switch to dark desk theme/i,
    })
    fireEvent.click(toggleBtn)
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
  })
})
