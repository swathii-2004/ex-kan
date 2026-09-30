import { Fragment } from 'react'

interface HighlightedTextProps {
  text: string
  words: string[]
  scores: number[]
  prediction: 'distress' | 'not-distress'
  method?: string
}

interface Span {
  content: string
  score?: number
}

/**
 * Highlights a method's (SHAP or LIME) top tokens inline within the original text.
 * These tokens are exact substrings of `text` (subword-level, sometimes with
 * leading/trailing whitespace attached), but /explain returns them unordered — no
 * offsets, and not necessarily adjacent in ranked-by-importance order either. So
 * this reconstructs whole words from the ORIGINAL TEXT first (splitting on
 * whitespace), then colors each whole word if any of its subword fragments
 * (e.g. "ag" + "ti" + "de", in any order, even with an unscored fragment like
 * "de" itself sitting in between) appear as a substring of it — one clean
 * highlight per word, using the strongest-magnitude matching fragment's score.
 * A word containing no matching fragment (or a token appearing in more than one
 * word) is handled the same way every occurrence is: matched independently.
 */
function buildSpans(text: string, words: string[], scores: number[]): Span[] {
  const candidates = words
    .map((word, i) => ({ word: word.trim(), score: scores[i] }))
    .filter((c) => c.word.length > 0)

  const chunks = text.match(/\S+|\s+/g) ?? []

  return chunks.map((chunk) => {
    if (/^\s/.test(chunk)) {
      return { content: chunk }
    }
    let bestScore: number | undefined
    for (const c of candidates) {
      if (chunk.includes(c.word) && (bestScore === undefined || Math.abs(c.score) > Math.abs(bestScore))) {
        bestScore = c.score
      }
    }
    return bestScore === undefined ? { content: chunk } : { content: chunk, score: bestScore }
  })
}

/**
 * The same merged whole-word spans HighlightedText renders, but as plain
 * (word, score) pairs — for anything that needs to talk about "the words that
 * mattered" (e.g. the plain-English summary sentence) without re-splitting
 * subword fragments like "ag"/"ti"/"tum" that buildSpans already reassembled
 * into "agtide" for display.
 */
export function getMergedWordScores(text: string, words: string[], scores: number[]): { word: string; score: number }[] {
  return buildSpans(text, words, scores)
    .filter((span): span is Required<Span> => span.score !== undefined)
    .map((span) => ({ word: span.content.trim(), score: span.score }))
    .filter((pair) => pair.word.length > 0)
}

export function HighlightedText({ text, words, scores, prediction, method = 'SHAP' }: HighlightedTextProps) {
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
            title={`${method} score: ${span.score.toFixed(4)}`}
          >
            {span.content}
          </mark>
        )
      })}
    </p>
  )
}
