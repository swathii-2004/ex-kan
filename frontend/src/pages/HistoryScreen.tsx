import { useCallback, useEffect, useState, type MouseEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { CodeMixBadge } from '../components/CodeMixBadge'
import {
  clearHistory,
  deleteHistoryItem,
  fetchHistory,
  fetchHistoryItem,
  type HistoryItem,
} from '../api/client'
import { useAnalysis } from '../context/AnalysisContext'

const PAGE_SIZE = 20

function snippet(text: string, maxLen = 80): string {
  const trimmed = text.trim().replace(/\s+/g, ' ')
  return trimmed.length > maxLen ? `${trimmed.slice(0, maxLen - 1)}…` : trimmed
}

function formatTime(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

export function HistoryScreen() {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busyId, setBusyId] = useState<string | null>(null)
  const { setResult } = useAnalysis()
  const navigate = useNavigate()

  const load = useCallback(async (skip: number, replace: boolean) => {
    try {
      const res = await fetchHistory(PAGE_SIZE, skip)
      setItems((prev) => (replace ? res.items : [...prev, ...res.items]))
      setTotal(res.total)
      setError(null)
    } catch {
      setError('Could not reach the history backend or database. It may be offline.')
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    fetchHistory(PAGE_SIZE, 0)
      .then((res) => {
        if (cancelled) return
        setItems(res.items)
        setTotal(res.total)
        setError(null)
      })
      .catch(() => {
        if (!cancelled) setError('Could not reach the history backend or database. It may be offline.')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function handleLoadMore() {
    setLoadingMore(true)
    await load(items.length, false)
    setLoadingMore(false)
  }

  async function handleRetry() {
    setLoading(true)
    await load(0, true)
    setLoading(false)
  }

  async function handleRowClick(id: string) {
    try {
      const detail = await fetchHistoryItem(id)
      setResult({
        text: detail.text,
        predict: {
          prediction: detail.prediction,
          confidence: detail.confidence,
          code_mix_bucket: detail.code_mix_bucket,
          effective_score: detail.effective_score,
        },
        explain: {
          shap_top_words: detail.shap_top_words,
          shap_scores: detail.shap_scores,
          lime_top_words: detail.lime_top_words,
          lime_scores: detail.lime_scores,
          lime_faithfulness_score: detail.lime_faithfulness_score,
          shap_faithfulness_score: detail.shap_faithfulness_score,
          saved: true,
          history_id: detail.id,
        },
      })
      navigate('/result')
    } catch {
      setError('Could not load that analysis. It may have been deleted.')
    }
  }

  async function handleDelete(e: MouseEvent, id: string) {
    e.stopPropagation()
    if (!window.confirm('Delete this analysis?')) return
    setBusyId(id)
    try {
      await deleteHistoryItem(id)
      setItems((prev) => prev.filter((it) => it.id !== id))
      setTotal((t) => t - 1)
    } catch {
      setError('Could not delete that analysis.')
    } finally {
      setBusyId(null)
    }
  }

  async function handleClearAll() {
    if (!window.confirm('Delete ALL history? This cannot be undone.')) return
    try {
      await clearHistory()
      setItems([])
      setTotal(0)
    } catch {
      setError('Could not clear history.')
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>History</h1>
        <p>Past analyses</p>
      </div>

      <p className="history-note">Texts you analyze are saved to the project database.</p>

      {error && (
        <div className="error-banner">
          {error}{' '}
          <button type="button" className="btn-toggle" onClick={handleRetry}>
            Retry
          </button>
        </div>
      )}

      {loading && !error && <p className="loading-text">Loading history…</p>}

      {!loading && !error && items.length === 0 && (
        <p className="empty-text">No analyses yet. Run one on the Analyze page.</p>
      )}

      {!loading && items.length > 0 && (
        <>
          <div className="history-toolbar">
            <button type="button" className="btn-toggle" onClick={handleClearAll}>
              Clear all history
            </button>
          </div>

          <ul className="history-list">
            {items.map((item) => {
              const combined = (item.lime_faithfulness_score + item.shap_faithfulness_score) / 2
              return (
                <li
                  key={item.id}
                  className="history-row card"
                  onClick={() => handleRowClick(item.id)}
                  role="button"
                  tabIndex={0}
                >
                  <div className="history-row__main">
                    <span className="history-row__time mono">{formatTime(item.created_at)}</span>
                    <span className="history-row__snippet">{snippet(item.text)}</span>
                  </div>
                  <div className="history-row__meta">
                    <span
                      className={`history-row__prediction history-row__prediction--${item.prediction}`}
                    >
                      {item.prediction === 'distress' ? 'Distress' : 'Not Distress'}
                    </span>
                    <CodeMixBadge bucket={item.code_mix_bucket} effectiveScore={item.effective_score} />
                    <span className="mono history-row__faithfulness">avg faithfulness {combined.toFixed(2)}</span>
                    <span className="mono history-row__version">{item.model_version}</span>
                    <button
                      type="button"
                      className="btn-toggle"
                      disabled={busyId === item.id}
                      onClick={(e) => handleDelete(e, item.id)}
                    >
                      Delete
                    </button>
                  </div>
                </li>
              )
            })}
          </ul>

          {items.length < total && (
            <div className="toggle-row">
              <button type="button" className="btn-toggle" disabled={loadingMore} onClick={handleLoadMore}>
                {loadingMore ? 'Loading…' : 'Load more'}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
