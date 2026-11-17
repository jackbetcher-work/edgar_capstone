import type { ReactNode } from 'react'
import {
  splitTextByInlineCitations,
  type ParsedInlineCitationToken,
  type ParsedInlineXbrlToken,
} from '../utils/formatters'

interface GroundedMarkdownProps {
  turnId: string
  markdown: string
  activeChunkId: string | null
  onSelectCitation: (token: ParsedInlineCitationToken) => void
  onSelectXbrlCitation?: (token: ParsedInlineXbrlToken) => void
}

function renderFormattedInlineText(text: string, keyPrefix: string): ReactNode[] {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g
  const parts: ReactNode[] = []
  let lastIdx = 0
  let match: RegExpExecArray | null

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(text.slice(lastIdx, match.index))
    }
    const token = match[0]
    const k = `${keyPrefix}-${match.index}`
    if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(
        <strong key={k} className="font-semibold text-ink-primary">
          {token.slice(2, -2)}
        </strong>
      )
    } else if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code
          key={k}
          className="tabular-nums-mono text-[11.5px] px-1 py-0.2 surface-subtle border border-hairline text-ink-primary"
        >
          {token.slice(1, -1)}
        </code>
      )
    } else if (token.startsWith('*') && token.endsWith('*')) {
      parts.push(
        <em key={k} className="italic text-ink-secondary">
          {token.slice(1, -1)}
        </em>
      )
    }
    lastIdx = pattern.lastIndex
  }

  if (lastIdx < text.length) {
    parts.push(text.slice(lastIdx))
  }
  return parts
}

function renderLineWithCitations(
  line: string,
  turnId: string,
  lineIdx: number,
  activeChunkId: string | null,
  onSelectCitation: (token: ParsedInlineCitationToken) => void,
  onSelectXbrlCitation?: (token: ParsedInlineXbrlToken) => void
): ReactNode[] {
  const segments = splitTextByInlineCitations(line)
  const nodes: ReactNode[] = []

  segments.forEach((seg, segIdx) => {
    if (seg.type === 'text') {
      nodes.push(
        ...renderFormattedInlineText(seg.value, `${turnId}-l${lineIdx}-s${segIdx}`)
      )
    } else if (seg.type === 'xbrl') {
      const { token } = seg
      const shortConcept = token.xbrlConcept.includes(':')
        ? token.xbrlConcept.split(':')[1]
        : token.xbrlConcept
      nodes.push(
        <button
          key={`${turnId}-xbrl-${lineIdx}-${segIdx}-${token.accessionNumber}`}
          id={`inline-xbrl-${turnId}-${lineIdx}-${segIdx}`}
          type="button"
          onClick={() => {
            if (onSelectXbrlCitation) {
              onSelectXbrlCitation(token)
            } else if (token.url) {
              window.open(token.url, '_blank', 'noreferrer')
            }
          }}
          className="inline-flex items-center gap-0.5 mx-0.5 px-1 py-0.2 text-[10px] tabular-nums-mono badge-accent hover:underline cursor-pointer align-baseline"
          title={`Inspect XBRL fact ${token.ticker} FY${token.fiscalYear} · ${token.xbrlConcept} · acc:${token.accessionNumber}`}
        >
          <span>
            [{token.ticker} FY{String(token.fiscalYear).slice(-2)} · XBRL {shortConcept}]
          </span>
        </button>
      )
    } else {
      const { token } = seg
      const isActive = activeChunkId === token.chunkId
      const shortSec = token.sectionLabel.toLowerCase().includes('1a')
        ? '1A'
        : '7'
      nodes.push(
        <button
          key={`${turnId}-cit-${lineIdx}-${segIdx}-${token.chunkId}`}
          id={`inline-cit-${turnId}-${lineIdx}-${segIdx}`}
          type="button"
          onClick={() => onSelectCitation(token)}
          className={`citation-chip ${isActive ? 'citation-chip-active' : ''}`}
          title={`Inspect ${token.ticker} FY${token.fiscalYear} ${token.sectionLabel} (${token.chunkId}) in Source Viewer`}
        >
          <span>
            [{token.ticker} FY{String(token.fiscalYear).slice(-2)} · ITEM {shortSec}]
          </span>
        </button>
      )
    }
  })

  return nodes
}

export function GroundedMarkdown({
  turnId,
  markdown,
  activeChunkId,
  onSelectCitation,
  onSelectXbrlCitation,
}: GroundedMarkdownProps) {
  const rawLines = markdown.split('\n')
  const blocks: ReactNode[] = []
  let bulletBuffer: Array<{ text: string; idx: number }> = []

  const flushBullets = (keyIdx: number) => {
    if (bulletBuffer.length === 0) return
    blocks.push(
      <ul
        key={`${turnId}-ul-${keyIdx}`}
        className="space-y-2 my-2 pl-4 list-square marker:text-[var(--bb-amber)]"
      >
        {bulletBuffer.map((item) => (
          <li
            key={`${turnId}-li-${item.idx}`}
            className="text-[12.5px] leading-relaxed text-ink-secondary pl-0.5"
          >
            {renderLineWithCitations(
              item.text,
              turnId,
              item.idx,
              activeChunkId,
              onSelectCitation,
              onSelectXbrlCitation
            )}
          </li>
        ))}
      </ul>
    )
    bulletBuffer = []
  }

  rawLines.forEach((rawLine, idx) => {
    const trimmed = rawLine.trim()
    if (!trimmed) {
      flushBullets(idx)
      return
    }

    if (trimmed.startsWith('> ')) {
      flushBullets(idx)
      const body = trimmed.slice(2)
      const isWarning = body.toLowerCase().includes('warning')
      blocks.push(
        <blockquote
          key={`${turnId}-bq-${idx}`}
          className={`my-1.5 px-2.5 py-1.5 text-[11.5px] leading-relaxed border-l-2 surface-subtle ${
            isWarning
              ? 'border-l-[var(--bb-amber)] text-ink-primary'
              : 'border-l-[var(--bb-cyan)] text-ink-secondary'
          }`}
        >
          {renderLineWithCitations(
            body,
            turnId,
            idx,
            activeChunkId,
            onSelectCitation,
            onSelectXbrlCitation
          )}
        </blockquote>
      )
      return
    }

    if (trimmed.startsWith('#### ')) {
      flushBullets(idx)
      blocks.push(
        <h4
          key={`${turnId}-h4-${idx}`}
          className="tabular-nums-mono text-[11px] font-semibold uppercase tracking-wider text-ink-muted mt-3 mb-1"
        >
          {trimmed.slice(5)}
        </h4>
      )
      return
    }

    if (trimmed.startsWith('### ')) {
      flushBullets(idx)
      blocks.push(
        <h3
          key={`${turnId}-h3-${idx}`}
          className="tabular-nums-mono text-xs font-semibold uppercase tracking-wider text-bb-amber mt-3 first:mt-0 pb-1 mb-1.5 border-b border-hairline"
        >
          {trimmed.slice(4)}
        </h3>
      )
      return
    }

    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      bulletBuffer.push({ text: trimmed.slice(2), idx })
      return
    }

    flushBullets(idx)
    blocks.push(
      <p
        key={`${turnId}-p-${idx}`}
        className="text-[12.5px] leading-relaxed text-ink-secondary my-1.5"
      >
        {renderLineWithCitations(
          trimmed,
          turnId,
          idx,
          activeChunkId,
          onSelectCitation,
          onSelectXbrlCitation
        )}
      </p>
    )
  })

  flushBullets(rawLines.length)

  return <div className="space-y-1">{blocks}</div>
}
