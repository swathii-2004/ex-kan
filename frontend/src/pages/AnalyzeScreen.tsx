import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { explain, predict } from '../api/client'
import { useAnalysis } from '../context/AnalysisContext'

export function AnalyzeScreen() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { setResult } = useAnalysis()
  const navigate = useNavigate()

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!text.trim() || loading) return

    setLoading(true)
    setError(null)
    try {
      const predictResponse = await predict(text)
      const explainResponse = await explain(text)
      setResult({ text, predict: predictResponse, explain: explainResponse })
      navigate('/result')
    } catch {
      setError('Something went wrong reaching the analysis backend. Is the API server running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>Analyze</h1>
        <p>Type a Kannada-English sentence to see what the model predicts, and why.</p>
      </div>

      <form className="analyze-form card" onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. Super movie guru, tumba chennagide but the story is weak"
        />
        <div className="analyze-form__footer">
          <button type="submit" className="btn-primary" disabled={loading || !text.trim()}>
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>
        {loading && (
          <p className="loading-text" style={{ marginTop: '0.75rem' }}>
            Running SHAP + LIME — this can take up to ~30 seconds for longer text.
          </p>
        )}
        {error && <div className="error-banner">{error}</div>}
      </form>
    </div>
  )
}
