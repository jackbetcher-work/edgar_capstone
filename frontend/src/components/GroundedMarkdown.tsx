import { useState, type ReactNode } from 'react'
import { ChevronDown, ChevronRight, Database, FileText } from 'lucide-react'
import {
  splitTextByInlineCitations,
  type ParsedInlineCitationToken,
  type ParsedInlineXbrlToken,
} from '../utils/formatters'
import { humanizeXbrlConcept, shortCitationTopic } from '../utils/xbrlLabels'

interface GroundedMarkdownProps {
  turnId: string
  markdown: string
  activeChunkId?: string | null
  /**
   * Optional matcher so chips citing a de-duplicated passage highlight together
   * with the merged source card; defaults to exact chunk-ID equality.
   */
  isChunkActive?: (chunkId: string) => boolean
  onSelectCitation: (token: ParsedInlineCitationToken) => void
  onSelectXbrlCitation?: (token: ParsedInlineXbrlToken) => void
}

interface RenderContext {
  turnId: string
  isChunkActive: (chunkId: string) => boolean
  onSelectCitation: (token: ParsedInlineCitationToken) => void
  onSelectXbrlCitation?: (token: ParsedInlineXbrlToken) => void
}

interface BulletItem {
  text: string
  idx: number
  children: Array<{ text: string; idx: number }>
}

/** Implementation-detail suffix stripped from section headings. */
const HEADING_NOISE_RE = /\s*\(deterministic(?: engine)?\)\s*$/i
/** Sub-bullet emitted under each finding: `*Evidence:* "verbatim quote"`. */
const EVIDENCE_PREFIX_RE = /^\*{0,2}Evidence\s*:\s*\*{0,2}\s*/i

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
  lineIdx: number,
  ctx: RenderContext
): ReactNode[] {
  const { turnId, isChunkActive, onSelectCitation, onSelectXbrlCitation } = ctx
  const segments = splitTextByInlineCitations(line)
  const nodes: ReactNode[] = []

  segments.forEach((seg, segIdx) => {
    if (seg.type === 'text') {
      nodes.push(
        ...renderFormattedInlineText(seg.value, `${turnId}-l${lineIdx}-s${segIdx}`)
      )
    } else if (seg.type === 'xbrl') {
      const { token } = seg
      const metricLabel = humanizeXbrlConcept(token.xbrlConcept)
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
          className="citation-chip"
          title={`Inspect XBRL fact ${token.ticker} FY${token.fiscalYear} · ${metricLabel} (${token.xbrlConcept}) · acc:${token.accessionNumber}`}
        >
          <Database className="w-2.5 h-2.5 shrink-0" aria-hidden="true" />
          <span>
            {token.ticker} FY{String(token.fiscalYear).slice(-2)} · {metricLabel}
          </span>
        </button>
      )
    } else {
      const { token } = seg
      const isActive = isChunkActive(token.chunkId)
      nodes.push(
        <button
          key={`${turnId}-cit-${lineIdx}-${segIdx}-${token.chunkId}`}
          id={`inline-cit-${turnId}-${lineIdx}-${segIdx}`}
          type="button"
          onClick={() => onSelectCitation(token)}
          className={`citation-chip ${isActive ? 'citation-chip-active' : ''}`}
          title={`Inspect ${token.ticker} FY${token.fiscalYear} · ${token.sectionLabel} in Source Viewer`}
        >
          <FileText className="w-2.5 h-2.5 shrink-0" aria-hidden="true" />
          <span>
            {token.ticker} FY{String(token.fiscalYear).slice(-2)} ·{' '}
            {shortCitationTopic(token.sectionLabel)}
          </span>
        </button>
      )
    }
  })

  return nodes
}

/** Verbatim supporting quotes under a finding, collapsed until requested. */
function EvidenceQuotes({
  quotes,
  itemKey,
  ctx,
}: {
  quotes: Array<{ text: string; idx: number }>
  itemKey: string
  ctx: RenderContext
}) {
  const [open, setOpen] = useState(false)
  const label = quotes.length === 1 ? 'Supporting quote' : `${quotes.length} supporting quotes`
  return (
    <div className="mt-0.5">
      <button
        id={`btn-evidence-${itemKey}`}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="inline-flex items-center gap-0.5 text-[10.5px] tabular-nums-mono uppercase tracking-wide text-ink-muted hover:text-ink-primary cursor-pointer"
      >
        {open ? (
          <ChevronDown className="w-3 h-3" aria-hidden="true" />
        ) : (
          <ChevronRight className="w-3 h-3" aria-hidden="true" />
        )}
        {label}
      </button>
      {open &&
        quotes.map((q) => (
          <blockquote
            key={`${itemKey}-q-${q.idx}`}
            className="mt-1 px-2.5 py-1.5 text-[11.5px] leading-relaxed border-l-2 border-l-[var(--border-strong)] surface-subtle text-ink-secondary italic"
          >
            {renderLineWithCitations(q.text.replace(EVIDENCE_PREFIX_RE, ''), q.idx, ctx)}
          </blockquote>
        ))}
    </div>
  )
}

function cleanHeading(text: string): string {
  return text.replace(HEADING_NOISE_RE, '').trim()
}

export function GroundedMarkdown({
  turnId,
  markdown,
  activeChunkId,
  isChunkActive,
  onSelectCitation,
  onSelectXbrlCitation,
}: GroundedMarkdownProps) {
  const ctx: RenderContext = {
    turnId,
    isChunkActive: isChunkActive ?? ((chunkId) => chunkId === activeChunkId),
    onSelectCitation,
    onSelectXbrlCitation,
  }
  const rawLines = markdown.split('\n')
  const blocks: ReactNode[] = []
  let bulletBuffer: BulletItem[] = []

  const flushBullets = (keyIdx: number) => {
    if (bulletBuffer.length === 0) return
    blocks.push(
      <ul
        key={`${turnId}-ul-${keyIdx}`}
        className="space-y-2 my-2 pl-4 list-square marker:text-[var(--bb-amber)]"
      >
        {bulletBuffer.map((item) => {
          const evidence = item.children.filter((c) => EVIDENCE_PREFIX_RE.test(c.text))
          const nested = item.children.filter((c) => !EVIDENCE_PREFIX_RE.test(c.text))
          const itemKey = `${turnId}-li-${item.idx}`
          return (
            <li
              key={itemKey}
              className="text-[12.5px] leading-relaxed text-ink-secondary pl-0.5"
            >
              {renderLineWithCitations(item.text, item.idx, ctx)}
              {nested.length > 0 && (
                <ul className="mt-1 space-y-1 pl-4 list-disc marker:text-[var(--text-muted)]">
                  {nested.map((child) => (
                    <li key={`${itemKey}-c-${child.idx}`}>
                      {renderLineWithCitations(child.text, child.idx, ctx)}
                    </li>
                  ))}
                </ul>
              )}
              {evidence.length > 0 && (
                <EvidenceQuotes quotes={evidence} itemKey={itemKey} ctx={ctx} />
              )}
            </li>
          )
        })}
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

    const isBullet = trimmed.startsWith('- ') || trimmed.startsWith('* ')
    const indent = rawLine.length - rawLine.trimStart().length
    if (isBullet && indent >= 2 && bulletBuffer.length > 0) {
      bulletBuffer[bulletBuffer.length - 1].children.push({
        text: trimmed.slice(2),
        idx,
      })
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
          {renderLineWithCitations(body, idx, ctx)}
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
          {cleanHeading(trimmed.slice(5))}
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
          {cleanHeading(trimmed.slice(4))}
        </h3>
      )
      return
    }

    if (isBullet) {
      bulletBuffer.push({ text: trimmed.slice(2), idx, children: [] })
      return
    }

    flushBullets(idx)
    blocks.push(
      <p
        key={`${turnId}-p-${idx}`}
        className="text-[12.5px] leading-relaxed text-ink-secondary my-1.5"
      >
        {renderLineWithCitations(trimmed, idx, ctx)}
      </p>
    )
  })

  flushBullets(rawLines.length)

  return <div className="space-y-1">{blocks}</div>
}
