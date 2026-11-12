import type { ReactNode } from 'react'
import { FileText } from 'lucide-react'
import {
  splitTextByInlineCitations,
  type ParsedInlineCitationToken,
} from '../utils/formatters'

interface GroundedMarkdownProps {
  turnId: string
  markdown: string
  activeChunkId: string | null
  onSelectCitation: (token: ParsedInlineCitationToken) => void
}

function renderFormattedInlineText(text: string, keyPrefix: string): ReactNode[] {
  // Parse **bold**, `code`, and *italic* cleanly without dangerouslySetInnerHTML
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
          className="tabular-nums-mono text-[12px] px-1.5 py-0.2 rounded surface-subtle text-ink-primary"
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
  onSelectCitation: (token: ParsedInlineCitationToken) => void
): ReactNode[] {
  const segments = splitTextByInlineCitations(line)
  const nodes: ReactNode[] = []

  segments.forEach((seg, segIdx) => {
    if (seg.type === 'text') {
      nodes.push(
        ...renderFormattedInlineText(seg.value, `${turnId}-l${lineIdx}-s${segIdx}`)
      )
    } else {
      const { token } = seg
      const isActive = activeChunkId === token.chunkId
      const shortSec = token.sectionLabel.toLowerCase().includes('1a')
        ? 'Item 1A'
        : 'Item 7'
      nodes.push(
        <button
          key={`${turnId}-cit-${lineIdx}-${segIdx}-${token.chunkId}`}
          id={`inline-cit-${turnId}-${lineIdx}-${segIdx}`}
          type="button"
          onClick={() => onSelectCitation(token)}
          className={`citation-chip ${isActive ? 'citation-chip-active' : ''}`}
          title={`Inspect ${token.ticker} FY${token.fiscalYear} ${token.sectionLabel} (${token.chunkId}) in Source Viewer`}
        >
          <FileText className="w-3 h-3 shrink-0" aria-hidden="true" />
          <span>
            {token.ticker} FY{String(token.fiscalYear).slice(-2)} · {shortSec}
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
}: GroundedMarkdownProps) {
  const rawLines = markdown.split('\n')
  const blocks: ReactNode[] = []
  let bulletBuffer: Array<{ text: string; idx: number }> = []

  const flushBullets = (keyIdx: number) => {
    if (bulletBuffer.length === 0) return
    blocks.push(
      <ul
        key={`${turnId}-ul-${keyIdx}`}
        className="space-y-2.5 my-2.5 pl-4 list-disc marker:text-[var(--text-muted)]"
      >
        {bulletBuffer.map((item) => (
          <li
            key={`${turnId}-li-${item.idx}`}
            className="text-[13.5px] leading-relaxed text-ink-secondary pl-1"
          >
            {renderLineWithCitations(
              item.text,
              turnId,
              item.idx,
              activeChunkId,
              onSelectCitation
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

    if (trimmed.startsWith('#### ')) {
      flushBullets(idx)
      blocks.push(
        <h4
          key={`${turnId}-h4-${idx}`}
          className="text-xs font-semibold uppercase tracking-wider text-ink-muted mt-4 mb-1.5"
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
          className="text-sm font-semibold tracking-tight text-ink-primary mt-4 first:mt-0 mb-1.5"
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
        className="text-[13.5px] leading-relaxed text-ink-secondary my-2"
      >
        {renderLineWithCitations(
          trimmed,
          turnId,
          idx,
          activeChunkId,
          onSelectCitation
        )}
      </p>
    )
  })

  flushBullets(rawLines.length)

  return <div className="space-y-1">{blocks}</div>
}
