/**
 * Parser for the Markdown passages stored in the 10-K vector corpus.
 *
 * Chunk text is produced by `vector_db/chunker.py` from EDGAR HTML → Markdown
 * conversions, so it carries artifacts that read poorly when shown verbatim:
 * - a `[TICKER (Company) | FY2024 Form 10-K | Section]` context header line
 * - backslash escapes on punctuation (`10\-K`, `U\.S\.`)
 * - running page footers (`Apple Inc\. | 2024 Form 10\-K | 21`)
 * - `•` bullets separated by blank lines
 * - SEC pipe tables whose `$` / `%` / `)` glyphs sit in their own cells
 *
 * `parseFilingMarkdown` turns that into a small block model that the
 * `FilingMarkdown` component renders as structured, readable evidence.
 */

export type FilingBlock =
  | { kind: 'heading'; level: 2 | 3; text: string }
  | { kind: 'paragraph'; text: string }
  | { kind: 'list'; items: string[] }
  | { kind: 'table'; header: string[] | null; rows: FilingTableRow[] }

export interface FilingTableRow {
  cells: string[]
  /** Label-only row (e.g. "Gross margin:") that spans the whole table. */
  isSection: boolean
}

/** Placeholder for escaped emphasis markers so they are not parsed as emphasis. */
export const ESCAPED_ASTERISK = '\uE000'
export const ESCAPED_UNDERSCORE = '\uE001'

const CONTEXT_HEADER_RE = /^\s*\[[^\n]*?\|\s*FY\d{4}\s+Form\s+10-K\b[^\n]*\]\s*(?:\n+|$)/i
const PAGE_FOOTER_RE = /\|\s*(?:\d{4}\s+)?Form\s+10-?K\s*\|\s*\d{1,4}\s*$/i
const INLINE_PAGE_FOOTER_RE = /Form\s+10-?K\s+\d{1,4}$/i
const PAGE_NUMBER_RE = /^(?:page\s+)?\d{1,3}$/i
const TABLE_OF_CONTENTS_RE = /^table of contents$/i
const BULLET_RE = /^(?:[•●◦▪‣·]|[-–*])\s+/
const MD_HEADING_RE = /^(#{1,6})\s+(.+)$/
const SEPARATOR_CELL_RE = /^:?-{2,}:?$/
const NUMERIC_CELL_RE = /^[\s$€£(−–—-]*\d[\d.,]*\s*[%)]*\s*\)?$|^[—–-]$/

/** Remove the chunker's `[TICKER (...) | FYxxxx Form 10-K | Section]` header line. */
export function stripChunkContextHeader(text: string): string {
  return (text ?? '').replace(CONTEXT_HEADER_RE, '')
}

/**
 * Undo Markdown backslash escapes. Escaped `*` / `_` become placeholders
 * (restored by `restoreEscapedMarkers`) so they never trigger emphasis.
 */
export function unescapeFilingMarkdown(text: string): string {
  return (text ?? '')
    .replace(/\\\*/g, ESCAPED_ASTERISK)
    .replace(/\\_/g, ESCAPED_UNDERSCORE)
    .replace(/\\([\\`{}[\]()#+\-.!|&$%,'"<>~:;=?@^/])/g, '$1')
}

export function restoreEscapedMarkers(text: string): string {
  return text
    .replace(new RegExp(ESCAPED_ASTERISK, 'g'), '*')
    .replace(new RegExp(ESCAPED_UNDERSCORE, 'g'), '_')
}

function isNoiseLine(trimmed: string): boolean {
  if (!trimmed.startsWith('|') && PAGE_FOOTER_RE.test(trimmed)) return true
  if (trimmed.length <= 90 && INLINE_PAGE_FOOTER_RE.test(trimmed)) return true
  return PAGE_NUMBER_RE.test(trimmed) || TABLE_OF_CONTENTS_RE.test(trimmed)
}

/** Running page footer that the HTML→Markdown step emitted as a table row. */
function isFooterRow(cells: string[]): boolean {
  const filled = cells.filter((c) => c !== '')
  if (filled.length === 0 || filled.length > 3) return false
  return (
    /Form\s+10-?K/i.test(filled.join(' ')) &&
    /^\d{1,4}$/.test(filled[filled.length - 1])
  )
}

function splitTableRow(line: string): string[] {
  let body = line.trim()
  if (body.startsWith('|')) body = body.slice(1)
  if (body.endsWith('|')) body = body.slice(0, -1)
  return body.split('|').map((c) => c.trim())
}

/**
 * Collapse SEC table artifacts: drop empty spacer cells, attach a lone `$`
 * to the following value and a lone `%` / `)` to the preceding value.
 * The first (label) cell is always kept so columns stay aligned with headers.
 */
export function compactTableCells(cells: string[]): string[] {
  if (cells.length === 0) return []
  const [label, ...rest] = cells
  const merged: string[] = []
  let pendingPrefix = ''
  for (const cell of rest) {
    if (cell === '') continue
    if (cell === '$' || cell === '€' || cell === '£') {
      pendingPrefix = cell
      continue
    }
    if (/^(?:%|\)|%\))$/.test(cell)) {
      if (merged.length > 0) merged[merged.length - 1] += cell
      continue
    }
    merged.push(`${pendingPrefix}${cell}`)
    pendingPrefix = ''
  }
  return [label, ...merged]
}

function parseTable(lines: string[]): FilingBlock {
  const rawRows = lines.map(splitTableRow)
  const isSeparator = (row: string[]) =>
    row.length > 0 &&
    row.some((c) => SEPARATOR_CELL_RE.test(c)) &&
    row.every((c) => c === '' || SEPARATOR_CELL_RE.test(c))

  let headerCells: string[] | null = null
  const bodyRaw: string[][] = []
  for (let idx = 0; idx < rawRows.length; idx += 1) {
    const row = rawRows[idx]
    if (isSeparator(row)) continue
    if (isFooterRow(row)) continue
    if (idx === 0 && rawRows.length > 1 && isSeparator(rawRows[1])) {
      const compacted = compactTableCells(row)
      // A header holding only a label ("Gross margin percentage:") is a section row.
      if (compacted.length === 1 && compacted[0]) bodyRaw.push(compacted)
      else headerCells = compacted
      continue
    }
    bodyRaw.push(compactTableCells(row))
  }

  const hasHeaderContent = headerCells?.some((c) => c !== '') ?? false
  const colCount = Math.max(
    hasHeaderContent && headerCells ? headerCells.length : 0,
    ...bodyRaw.map((r) => r.length),
    1
  )
  const pad = (cells: string[]) =>
    cells.length >= colCount
      ? cells
      : [...cells, ...Array.from({ length: colCount - cells.length }, () => '')]

  const rows: FilingTableRow[] = bodyRaw
    .filter((cells) => cells.some((c) => c !== ''))
    .map((cells) => {
      const isSection = cells.length === 1 && colCount > 1
      return { cells: isSection ? cells : pad(cells), isSection }
    })

  return {
    kind: 'table',
    header: hasHeaderContent && headerCells ? pad(headerCells) : null,
    rows,
  }
}

function standaloneHeading(trimmed: string): { level: 2 | 3; text: string } | null {
  const bold = /^\*\*([^*]+)\*\*:?$/.exec(trimmed)
  if (bold && bold[1].trim().length > 0 && bold[1].length <= 160) {
    return { level: 2, text: bold[1].trim() }
  }
  const italic = /^\*([^*]+)\*$/.exec(trimmed)
  if (italic && italic[1].trim().length > 0 && italic[1].length <= 100) {
    return { level: 3, text: italic[1].trim() }
  }
  return null
}

/** Parse a raw corpus chunk (header, escapes, tables and all) into render blocks. */
export function parseFilingMarkdown(raw: string): FilingBlock[] {
  const text = unescapeFilingMarkdown(stripChunkContextHeader(raw))
  const lines = text.replace(/\r\n?/g, '\n').split('\n')
  const blocks: FilingBlock[] = []
  let paragraph: string[] = []
  let listItems: string[] = []

  const flushParagraph = () => {
    if (paragraph.length > 0) {
      blocks.push({ kind: 'paragraph', text: paragraph.join(' ') })
      paragraph = []
    }
  }
  const flushList = () => {
    if (listItems.length > 0) {
      blocks.push({ kind: 'list', items: listItems })
      listItems = []
    }
  }
  const nextContentLine = (from: number): string | null => {
    for (let j = from; j < lines.length; j += 1) {
      const t = lines[j].trim()
      if (t && !isNoiseLine(t)) return t
    }
    return null
  }

  for (let i = 0; i < lines.length; i += 1) {
    const trimmed = lines[i].trim()

    if (!trimmed) {
      flushParagraph()
      // 10-K bullets are usually separated by blank lines; keep the list open.
      if (listItems.length > 0) {
        const next = nextContentLine(i + 1)
        if (!next || !BULLET_RE.test(next)) flushList()
      }
      continue
    }
    if (isNoiseLine(trimmed)) continue

    if (trimmed.startsWith('|')) {
      flushParagraph()
      flushList()
      const tableLines: string[] = []
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        tableLines.push(lines[i])
        i += 1
      }
      i -= 1
      const table = parseTable(tableLines)
      if (table.kind === 'table' && (table.rows.length > 0 || table.header)) {
        blocks.push(table)
      }
      continue
    }

    const mdHeading = MD_HEADING_RE.exec(trimmed)
    if (mdHeading) {
      flushParagraph()
      flushList()
      blocks.push({
        kind: 'heading',
        level: mdHeading[1].length <= 3 ? 2 : 3,
        text: mdHeading[2].replace(/^\*\*(.+)\*\*$/, '$1').trim(),
      })
      continue
    }

    const heading = paragraph.length === 0 ? standaloneHeading(trimmed) : null
    if (heading) {
      flushList()
      blocks.push({ kind: 'heading', ...heading })
      continue
    }

    if (BULLET_RE.test(trimmed)) {
      flushParagraph()
      listItems.push(trimmed.replace(BULLET_RE, ''))
      continue
    }

    if (listItems.length > 0) flushList()
    paragraph.push(trimmed)
  }

  flushParagraph()
  flushList()
  return blocks
}

/** True when a table cell holds a figure that should be right-aligned. */
export function isNumericCell(cell: string): boolean {
  return NUMERIC_CELL_RE.test(cell.trim())
}

/** Strip inline emphasis markers for plain-text previews and comparisons. */
export function stripInlineMarkdown(text: string): string {
  return restoreEscapedMarkers(
    text
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/\*([^*]+)\*/g, '$1')
      .replace(/`([^`]+)`/g, '$1')
  )
}

/**
 * One-line plain-text preview of a passage for collapsed evidence cards.
 * Skips the leading heading when it just repeats the card's topic.
 */
export function filingPassagePreview(
  raw: string,
  topic: string | null = null,
  maxChars = 200
): string {
  const blocks = parseFilingMarkdown(raw)
  const topicNorm = (topic ?? '').trim().toLowerCase()
  let preview = ''
  for (const block of blocks) {
    if (block.kind === 'heading') {
      if (topicNorm && block.text.trim().toLowerCase() === topicNorm) continue
      preview = block.text
    } else if (block.kind === 'paragraph') {
      preview = block.text
    } else if (block.kind === 'list') {
      preview = block.items.join(' · ')
    } else {
      const label = block.rows.find((r) => r.cells[0])?.cells[0]
      preview = label ? `Table: ${label}…` : 'Table'
    }
    if (preview) break
  }
  const plain = stripInlineMarkdown(preview).replace(/\s+/g, ' ').trim()
  if (plain.length <= maxChars) return plain
  const clipped = plain.slice(0, maxChars).replace(/\s+\S*$/, '')
  return `${clipped}…`
}

/**
 * A paragraph from the passage suitable for locating it inside the full 10-K
 * section (the reader matches on the first ~95 cleaned characters of a block).
 */
export function filingHighlightAnchor(raw: string): string {
  const body = stripChunkContextHeader(raw)
  const paragraphs = body
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean)
  const prose = paragraphs.find(
    (p) => !p.startsWith('|') && !/^\*\*[^*]+\*\*$/.test(p) && p.length >= 60
  )
  return prose ?? paragraphs[0] ?? body
}
