import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fetchDashboardStats, type BucketStats } from '../api/client'

export function DashboardScreen() {
  const [buckets, setBuckets] = useState<BucketStats[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchDashboardStats()
      .then((res) => setBuckets(res.buckets))
      .catch(() => setError('Could not reach /dashboard-stats. Is the API server running?'))
  }, [])

  return (
    <div className="page">
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>Does trust in explanations hold up at scale?</p>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {!buckets && !error && <p className="loading-text">Loading real faithfulness results…</p>}

      {buckets && (
        <>
          <div className="metric-cards">
            {buckets.map((b) => (
              <div key={b.bucket} className="card metric-card">
                <div className="metric-card__bucket">{b.bucket}</div>
                <div className="metric-card__scores">
                  <span className="metric-card__lime">LIME {b.lime_avg.toFixed(2)}</span>
                  <span className="metric-card__shap">SHAP {b.shap_avg.toFixed(2)}</span>
                </div>
                <div className="metric-card__n">n={b.n}</div>
              </div>
            ))}
          </div>

          <div className="card chart-card">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={buckets} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="bucket" tickFormatter={(v) => String(v).toUpperCase()} stroke="var(--ink-soft)" />
                <YAxis domain={[0, 1]} stroke="var(--ink-soft)" />
                <Tooltip />
                <Legend />
                <Bar dataKey="lime_avg" name="LIME" fill="#1f5c52" radius={[4, 4, 0, 0]} />
                <Bar dataKey="shap_avg" name="SHAP" fill="#b5541e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="key-finding">
            <h3>Key Finding</h3>
            <p>
              Faithfulness follows a <strong>U-shaped pattern</strong>, not a monotonic
              decline: the Medium code-mix bucket shows the lowest faithfulness of the
              three, while Low and High are both higher and close to each other. This
              contradicts the original hypothesis that faithfulness degrades steadily
              as code-mixing increases — the pattern held up across explanation method
              (LIME and SHAP), faithfulness-threshold choice, and after correcting for
              a text-length confound in the deletion/insertion tests.
            </p>
          </div>
        </>
      )}
    </div>
  )
}
