const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export interface PredictResponse {
  prediction: 'distress' | 'not-distress'
  confidence: number
  code_mix_bucket: 'low' | 'medium' | 'high'
  effective_score: number
}

export interface ExplainResponse {
  shap_top_words: string[]
  shap_scores: number[]
  lime_top_words: string[]
  lime_scores: number[]
  lime_faithfulness_score: number
  shap_faithfulness_score: number
}

export interface BucketStats {
  bucket: 'low' | 'medium' | 'high'
  n: number
  lime_avg: number
  shap_avg: number
}

export interface DashboardStatsResponse {
  buckets: BucketStats[]
}

async function postText<T>(path: string, text: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export function predict(text: string): Promise<PredictResponse> {
  return postText<PredictResponse>('/predict', text)
}

export function explain(text: string): Promise<ExplainResponse> {
  return postText<ExplainResponse>('/explain', text)
}

export async function fetchDashboardStats(): Promise<DashboardStatsResponse> {
  const res = await fetch(`${API_BASE_URL}/dashboard-stats`)
  if (!res.ok) {
    throw new Error(`/dashboard-stats failed: ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<DashboardStatsResponse>
}
