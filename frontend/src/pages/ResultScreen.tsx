import { Link } from 'react-router-dom'
import { CodeMixBadge } from '../components/CodeMixBadge'
import { FaithfulnessGauge } from '../components/FaithfulnessGauge'
import { HighlightedText } from '../components/HighlightedText'
import { useAnalysis } from '../context/AnalysisContext'

export function ResultScreen() {
  const { result } = useAnalysis()

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

          <HighlightedText
            text={text}
            words={explain.shap_top_words}
            scores={explain.shap_scores}
            prediction={predict.prediction}
          />

          <div>
            <strong style={{ fontSize: '0.85rem', color: 'var(--ink-soft)' }}>Top LIME words</strong>
            <div className="word-list">
              {explain.lime_top_words.slice(0, 8).map((word, i) => (
                <span key={i} className="word-chip">
                  {word} ({explain.lime_scores[i] >= 0 ? '+' : ''}
                  {explain.lime_scores[i].toFixed(3)})
                </span>
              ))}
            </div>
          </div>
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
    </div>
  )
}
