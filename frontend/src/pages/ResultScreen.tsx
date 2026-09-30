import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeMixBadge } from '../components/CodeMixBadge'
import { FaithfulnessGauge } from '../components/FaithfulnessGauge'
import { getMergedWordScores, HighlightedText } from '../components/HighlightedText'
import { useAnalysis } from '../context/AnalysisContext'

function WordScoreList({ words, scores }: { words: string[]; scores: number[] }) {
  return (
    <div className="word-list">
      {words.slice(0, 8).map((word, i) => (
        <span key={i} className="word-chip">
          {word} ({scores[i] >= 0 ? '+' : ''}
          {scores[i].toFixed(3)})
        </span>
      ))}
    </div>
  )
}

function formatWordList(words: string[]): string {
  const quoted = words.map((w) => `"${w}"`)
  if (quoted.length <= 1) return quoted.join('')
  return `${quoted.slice(0, -1).join(', ')} and ${quoted[quoted.length - 1]}`
}

/**
 * Turns a method's ranked (word, score) pairs into a plain-English sentence: which
 * word drove the prediction most, which others agreed, and which word (if any)
 * pointed the other way. Positive score = pushes toward the shown prediction,
 * negative = pushes against it — same convention the highlight color/intensity uses.
 * Callers must pass the same merged whole-word spans HighlightedText displays
 * (via getMergedWordScores), not the raw top_words list — otherwise subword
 * fragments like "ag"/"ti"/"tum" show up instead of "agtide".
 */
function buildExplanationSummary(pairs: { word: string; score: number }[], predictionLabel: string): string | null {
  if (pairs.length === 0) return null

  const supporting = pairs.filter((p) => p.score > 0).sort((a, b) => b.score - a.score)
  const opposing = pairs.filter((p) => p.score < 0).sort((a, b) => a.score - b.score)

  let sentence: string
  if (supporting.length === 0) {
    sentence = `No individual word strongly pushed toward ${predictionLabel} — the prediction likely reflects overall sentence patterns rather than a few standout words.`
  } else {
    const [strongest, ...rest] = supporting
    const others = rest.slice(0, 2)
    const alsoClause = others.length > 0 ? `, along with ${formatWordList(others.map((o) => o.word))}` : ''
    const wordCount = 1 + others.length
    sentence = `The AI marked this as ${predictionLabel} mainly because of the word "${strongest.word}" (strongest signal)${alsoClause} — ${wordCount > 1 ? 'these words' : 'this word'} pushed the prediction toward ${predictionLabel}.`
  }

  if (opposing.length > 0) {
    const strongestOpposing = opposing[0]
    const referenceScore = supporting[0]?.score ?? Math.abs(strongestOpposing.score)
    const isSlight = Math.abs(strongestOpposing.score) < 0.5 * Math.abs(referenceScore)
    const extra = opposing.length > 1 ? ` (and ${opposing.length - 1} other word${opposing.length - 1 > 1 ? 's' : ''})` : ''
    sentence += ` The word "${strongestOpposing.word}"${extra} actually pushed ${isSlight ? 'slightly ' : ''}AWAY from ${predictionLabel}.`
  }

  return sentence
}

function ExplanationSummary({
  text,
  words,
  scores,
  predictionLabel,
}: {
  text: string
  words: string[]
  scores: number[]
  predictionLabel: string
}) {
  const mergedPairs = getMergedWordScores(text, words, scores)
  const summary = buildExplanationSummary(mergedPairs, predictionLabel)
  if (!summary) return null
  return <p className="explanation-summary">{summary}</p>
}

export function ResultScreen() {
  const { result } = useAnalysis()
  const [showDetails, setShowDetails] = useState(false)

  if (!result) {
    return (
      <div className="page">
        <div className="page-header">
          <h1>Result</h1>
        </div>
        <p className="empty-text">
          No analysis yet — head to <Link to="/">Analyze</Link> and submit some text first.
        </p>
      </div>
    )
  }

  const { text, predict, explain } = result
  const predictionLabel = predict.prediction === 'distress' ? 'Distress' : 'Not Distress'
  const labelClass =
    predict.prediction === 'distress' ? 'prediction-label--distress' : 'prediction-label--not-distress'

  return (
    <div className="page">
      <div className="page-header">
        <h1>Result</h1>
        <p>Can we trust why the model reacted the way it did?</p>
      </div>

      <div className="result-grid">
        <div className="card prediction-card">
          <CodeMixBadge bucket={predict.code_mix_bucket} effectiveScore={predict.effective_score} />
          <span className={`prediction-label ${labelClass}`}>{predictionLabel}</span>
          <span className="confidence-line">confidence {(predict.confidence * 100).toFixed(1)}%</span>
        </div>

        <div className="card">
          <h3 style={{ fontSize: '1rem', marginBottom: '1rem' }}>Faithfulness</h3>
          <FaithfulnessGauge
            limeScore={explain.lime_faithfulness_score}
            shapScore={explain.shap_faithfulness_score}
          />
          <p style={{ fontSize: '0.85rem', color: 'var(--ink-soft)', marginTop: '1rem' }}>
            Computed live for this sentence via deletion/insertion testing — not the
            aggregate bucket average shown on the Dashboard.
          </p>
        </div>
      </div>

      <div className="explanation-grid">
        <div className="card explanation-card">
          <h3 style={{ fontSize: '1rem' }}>SHAP Explanation</h3>
          <HighlightedText
            text={text}
            words={explain.shap_top_words}
            scores={explain.shap_scores}
            prediction={predict.prediction}
            method="SHAP"
          />
          <ExplanationSummary
            text={text}
            words={explain.shap_top_words}
            scores={explain.shap_scores}
            predictionLabel={predictionLabel}
          />
        </div>

        <div className="card explanation-card">
          <h3 style={{ fontSize: '1rem' }}>LIME Explanation</h3>
          <HighlightedText
            text={text}
            words={explain.lime_top_words}
            scores={explain.lime_scores}
            prediction={predict.prediction}
            method="LIME"
          />
          <ExplanationSummary
            text={text}
            words={explain.lime_top_words}
            scores={explain.lime_scores}
            predictionLabel={predictionLabel}
          />
        </div>
      </div>

      <div className="toggle-row">
        <button
          type="button"
          className="btn-toggle"
          onClick={() => setShowDetails((v) => !v)}
          aria-expanded={showDetails}
        >
          {showDetails ? '▾' : '▸'} Show detailed scores
        </button>
      </div>

      {showDetails && (
        <div className="explanation-grid">
          <div className="card explanation-card">
            <strong style={{ fontSize: '0.85rem', color: 'var(--ink-soft)' }}>SHAP word scores</strong>
            <WordScoreList words={explain.shap_top_words} scores={explain.shap_scores} />
          </div>

          <div className="card explanation-card">
            <strong style={{ fontSize: '0.85rem', color: 'var(--ink-soft)' }}>LIME word scores</strong>
            <WordScoreList words={explain.lime_top_words} scores={explain.lime_scores} />
          </div>
        </div>
      )}
    </div>
  )
}
