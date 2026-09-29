import { Fragment } from 'react'

interface HighlightedTextProps {
  text: string
  words: string[]
  scores: number[]
  prediction: 'distress' | 'not-distress'
}

interface Span {
  content: string
  score?: number
}

/**
 * Highlights SHAP's top tokens inline within the original text. SHAP's tokens are
 * exact substrings of `text` (subword-level, sometimes with trailing whitespace
 * attached), but /explain returns them unordered-by-position with no offsets — so
 * this does a greedy longest-match-first scan rather than relying on index data.
 * A token appearing more than once in the text will have all its occurrences
 * highlighted, which is a known simplification of this approach.
 */
function buildSpans(text: string, words: string[], scores: number[]): Span[] {
  const candidates = words
    .map((word, i) => ({ word, score: scores[i] }))
    .filter((c) => c.word.trim().length > 0)
    .sort((a, b) => b.word.length - a.word.length)

  const spans: Span[] = []
  let cursor = 0

  while (cursor < text.length) {
    const match = candidates.find((c) => text.startsWith(c.word, cursor))
    if (match) {
      spans.push({ content: match.word, score: match.score })
      cursor += match.word.length
    } else {
      const next = candidates
        .map((c) => text.indexOf(c.word, cursor))
        .filter((idx) => idx !== -1)
        .sort((a, b) => a - b)[0]
      const end = next === undefined ? text.length : next
      spans.push({ content: text.slice(cursor, end) })
      cursor = end
    }
  }

  return spans
}

export function HighlightedText({ text, words, scores, prediction }: HighlightedTextProps) {
  const spans = buildSpans(text, words, scores)
  const maxAbsScore = Math.max(0.0001, ...scores.map((s) => Math.abs(s)))
  const color = prediction === 'distress' ? '181, 84, 30' : '31, 92, 82'

  return (
    <p className="highlighted-text">
      {spans.map((span, i) => {
        if (span.score === undefined) {
          return <Fragment key={i}>{span.content}</Fragment>
        }
        const intensity = Math.min(1, Math.abs(span.score) / maxAbsScore)
        return (
          <mark
            key={i}
            style={{ backgroundColor: `rgba(${color}, ${0.15 + intensity * 0.45})` }}
            title={`SHAP score: ${span.score.toFixed(4)}`}
          >
            {span.content}
          </mark>
        )
      })}
    </p>
  )
}
