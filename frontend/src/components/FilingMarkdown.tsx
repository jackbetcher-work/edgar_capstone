import { useMemo, type ReactNode } from 'react'
import {
  isNumericCell,
  parseFilingMarkdown,
  restoreEscapedMarkers,
  type FilingBlock,
} from '../utils/filingMarkdown'

interface FilingMarkdownProps {
  /** Raw corpus chunk text (context header, escapes and pipe tables allowed). */
  markdown: string
  /** Leading heading to suppress when it only repeats the card title. */
  omitLeadingHeading?: string | null
  className?: string
}

const INLINE_RE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let last = 0
  let match: RegExpExecArray | null
  const re = new RegExp(INLINE_RE.source, 'g')
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) {
      nodes.push(restoreEscapedMarkers(text.slice(last, match.index)))
    }
    const token = match[0]
    const key = `${keyPrefix}-${match.index}`
    if (token.startsWith('**')) {
      nodes.push(
        <strong key={key} className="font-semibold text-ink-primary">
          {restoreEscapedMarkers(token.slice(2, -2).trim())}
        </strong>
      )
    } else if (token.startsWith('`')) {
      nodes.push(
        <code key={key} className="tabular-nums-mono text-[11px]">
          {restoreEscapedMarkers(token.slice(1, -1))}
        </code>
      )
    } else {
      nodes.push(
        <em key={key} className="italic">
          {restoreEscapedMarkers(token.slice(1, -1))}
        </em>
      )
    }
    last = re.lastIndex
  }
  if (last < text.length) nodes.push(restoreEscapedMarkers(text.slice(last)))
  return nodes
}

function renderBlock(block: FilingBlock, idx: number): ReactNode {
  const key = `fb-${idx}`
  switch (block.kind) {
    case 'heading':
      return block.level === 2 ? (
        <h5
          key={key}
          className="text-[12px] font-semibold text-ink-primary mt-2.5 first:mt-0"
        >
          {renderInline(block.text, key)}
        </h5>
      ) : (
        <h6
          key={key}
          className="text-[11.5px] font-semibold italic text-ink-secondary mt-2 first:mt-0"
        >
          {renderInline(block.text, key)}
        </h6>
      )
    case 'paragraph':
      return (
        <p key={key} className="my-1.5 first:mt-0">
          {renderInline(block.text, key)}
        </p>
      )
    case 'list':
      return (
        <ul
          key={key}
          className="my-1.5 pl-4 space-y-0.5 list-disc marker:text-[var(--text-muted)]"
        >
          {block.items.map((item, i) => (
            <li key={`${key}-${i}`}>{renderInline(item, `${key}-${i}`)}</li>
          ))}
        </ul>
      )
    case 'table': {
      const colCount = block.header?.length ?? block.rows[0]?.cells.length ?? 1
      return (
        <div key={key} className="my-2 overflow-x-auto border border-hairline">
          <table className="w-full border-collapse text-[11px] tabular-nums-mono">
            {block.header && (
              <thead>
                <tr className="surface-subtle border-b border-hairline text-ink-muted">
                  {block.header.map((cell, ci) => (
                    <th
                      key={`${key}-h-${ci}`}
                      className={`px-2 py-1 font-medium whitespace-nowrap ${
                        ci === 0 ? 'text-left' : 'text-right'
                      }`}
                    >
                      {renderInline(cell, `${key}-h-${ci}`)}
                    </th>
                  ))}
                </tr>
              </thead>
            )}
            <tbody className="divide-y divide-[var(--border-hairline)]">
              {block.rows.map((row, ri) =>
                row.isSection ? (
                  <tr key={`${key}-r-${ri}`}>
                    <td
                      colSpan={colCount}
                      className="px-2 py-1 text-[10.5px] font-semibold uppercase tracking-wide text-ink-muted"
                    >
                      {renderInline(row.cells[0], `${key}-r-${ri}`)}
                    </td>
                  </tr>
                ) : (
                  <tr key={`${key}-r-${ri}`}>
                    {row.cells.map((cell, ci) => (
                      <td
                        key={`${key}-r-${ri}-${ci}`}
                        className={`px-2 py-1 align-top ${
                          ci === 0
                            ? 'text-left font-sans text-ink-secondary'
                            : isNumericCell(cell)
                              ? 'text-right whitespace-nowrap text-ink-primary'
                              : 'text-left font-sans text-ink-secondary'
                        }`}
                      >
                        {renderInline(cell, `${key}-r-${ri}-${ci}`)}
                      </td>
                    ))}
                  </tr>
                )
              )}
            </tbody>
          </table>
        </div>
      )
    }
  }
}

/** Structured rendering of a 10-K corpus passage (headings, bullets, tables). */
export function FilingMarkdown({
  markdown,
  omitLeadingHeading = null,
  className = '',
}: FilingMarkdownProps) {
  const blocks = useMemo(() => {
    const parsed = parseFilingMarkdown(markdown)
    const omit = omitLeadingHeading?.trim().toLowerCase()
    const first = parsed[0]
    if (omit && first?.kind === 'heading' && first.text.trim().toLowerCase() === omit) {
      return parsed.slice(1)
    }
    return parsed
  }, [markdown, omitLeadingHeading])

  return (
    <div className={`filing-prose text-[12.5px] leading-relaxed ${className}`}>
      {blocks.map(renderBlock)}
    </div>
  )
}
